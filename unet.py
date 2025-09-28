import os
import time
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.mask import mask
from rasterio.plot import show
from shapely.geometry import box
from rasterio.windows import Window
from rasterio.transform import from_bounds

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Verificar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

class AdaptiveSuperResTiffDataset(Dataset):
    """
    Dataset adaptativo que permite diferentes resoluções de saída
    """
    def __init__(self, low_res_files, high_res_files, target_resolution=None, transform=None, patch_size=128):
        self.low_res_datasets = [rasterio.open(f) for f in low_res_files]
        self.high_res_datasets = [rasterio.open(f) for f in high_res_files]
        self.transform = transform
        self.patch_size = patch_size
        self.target_resolution = target_resolution  # Em metros (ex: 0.5, 5.0, 10.0)
        
        assert len(self.low_res_datasets) == len(self.high_res_datasets), \
            "Número de arquivos low-res e high-res deve ser igual"
        
        # Calcular fator de escala baseado nas resoluções dos arquivos
        self._calculate_resolution_info()
    
    def _calculate_resolution_info(self):
        """Calcula informações de resolução dos dados"""
        self.resolution_info = []
        
        for low_src, high_src in zip(self.low_res_datasets, self.high_res_datasets):
            # Obter resolução em metros dos pixels
            low_res_x = abs(low_src.transform.a)  # resolução X em metros
            low_res_y = abs(low_src.transform.e)  # resolução Y em metros
            
            high_res_x = abs(high_src.transform.a)
            high_res_y = abs(high_src.transform.e)
            
            # Calcular fator de escala
            scale_x = low_res_x / high_res_x
            scale_y = low_res_y / high_res_y
            
            info = {
                'low_res': (low_res_x, low_res_y),
                'high_res': (high_res_x, high_res_y),
                'scale_factor': (scale_x, scale_y),
                'avg_scale': (scale_x + scale_y) / 2
            }
            
            self.resolution_info.append(info)
            
            print(f"Arquivo {len(self.resolution_info)}: Low-res: {low_res_x:.2f}x{low_res_y:.2f}m, "
                  f"High-res: {high_res_x:.2f}x{high_res_y:.2f}m, "
                  f"Fator de escala: {info['avg_scale']:.2f}x")
    
    def get_target_resolution_factor(self, current_resolution):
        """
        Calcula o fator de escala necessário para atingir a resolução alvo
        """
        if self.target_resolution is None:
            return 1.0
        
        return current_resolution / self.target_resolution
    
    def __len__(self):
        return 10000  # Número virtual de patches para treinamento
    
    def __getitem__(self, idx):
        # Selecionar arquivo aleatoriamente
        file_idx = np.random.randint(0, len(self.low_res_datasets))
        low_src = self.low_res_datasets[file_idx]
        high_src = self.high_res_datasets[file_idx]
        
        # Verificar se as imagens são grandes o suficiente
        max_x = min(low_src.width, high_src.width) - self.patch_size
        max_y = min(low_src.height, high_src.height) - self.patch_size
        
        if max_x <= 0 or max_y <= 0:
            # Se muito pequena, usar imagem inteira
            window = Window(0, 0, min(low_src.width, high_src.width), 
                          min(low_src.height, high_src.height))
        else:
            # Selecionar patch aleatório
            random_x = np.random.randint(0, max_x)
            random_y = np.random.randint(0, max_y)
            window = Window(random_x, random_y, self.patch_size, self.patch_size)
        
        # Ler dados
        try:
            img_low = low_src.read(1, window=window).astype(np.float32)
            img_high = high_src.read(1, window=window).astype(np.float32)
        except Exception as e:
            print(f"Erro ao ler patch: {e}")
            # Fallback para patch menor
            img_low = low_src.read(1).astype(np.float32)
            img_high = high_src.read(1).astype(np.float32)
        
        # Normalização robusta
        img_low_norm = self._normalize_image(img_low, low_src.nodata)
        img_high_norm = self._normalize_image(img_high, high_src.nodata)
        
        # Converter para tensores
        low_tensor = torch.from_numpy(img_low_norm).unsqueeze(0)
        high_tensor = torch.from_numpy(img_high_norm).unsqueeze(0)
        
        # Aplicar transformações se especificadas
        if self.transform:
            low_tensor = self.transform(low_tensor)
            high_tensor = self.transform(high_tensor)
        
        # Adicionar informação de resolução como metadado
        resolution_factor = self.resolution_info[file_idx]['avg_scale']
        
        return low_tensor, high_tensor, resolution_factor
    
    def _normalize_image(self, img, nodata_value):
        """Normaliza imagem de forma robusta"""
        if nodata_value is not None:
            valid_mask = img != nodata_value
        else:
            valid_mask = np.isfinite(img)
        
        if not np.any(valid_mask):
            return np.zeros_like(img)
        
        valid_data = img[valid_mask]
        
        # Usar percentis para normalização mais robusta
        p2, p98 = np.percentile(valid_data, [2, 98])
        
        if p98 - p2 > 0:
            img_norm = np.zeros_like(img)
            img_norm[valid_mask] = np.clip((img[valid_mask] - p2) / (p98 - p2), 0, 1)
            return img_norm
        else:
            return np.zeros_like(img)
    
    def __del__(self):
        """Fechar todos os datasets ao destruir o objeto"""
        for dataset in self.low_res_datasets + self.high_res_datasets:
            if hasattr(dataset, 'close'):
                dataset.close()


class ResolutionAwareUNet(nn.Module):
    """
    U-Net adaptativa com consciência de resolução
    """
    def __init__(self, in_channels=1, out_channels=1, base_filters=32, num_resolution_levels=3):
        super().__init__()
        
        self.num_resolution_levels = num_resolution_levels
        f = base_filters
        
        # Encoder
        self.conv1 = self._conv_block(in_channels, f)
        self.conv2 = self._conv_block(f, f*2)
        self.conv3 = self._conv_block(f*2, f*4)
        self.conv4 = self._conv_block(f*4, f*8)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Bottleneck com atenção
        self.bottleneck = nn.Sequential(
            self._conv_block(f*8, f*16),
            nn.AdaptiveAvgPool2d(1),  # Global Average Pooling
            nn.Conv2d(f*16, f*16//4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f*16//4, f*16, 1),
            nn.Sigmoid()
        )
        
        self.bottleneck_conv = self._conv_block(f*8, f*16)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(f*16, f*8, kernel_size=2, stride=2)
        self.conv_up4 = self._conv_block(f*16, f*8)
        
        self.up3 = nn.ConvTranspose2d(f*8, f*4, kernel_size=2, stride=2)
        self.conv_up3 = self._conv_block(f*8, f*4)
        
        self.up2 = nn.ConvTranspose2d(f*4, f*2, kernel_size=2, stride=2)
        self.conv_up2 = self._conv_block(f*4, f*2)
        
        self.up1 = nn.ConvTranspose2d(f*2, f, kernel_size=2, stride=2)
        self.conv_up1 = self._conv_block(f*2, f)
        
        # Saída final com múltiplas escalas
        self.final_conv = nn.Conv2d(f, out_channels, kernel_size=1)
        
        # Camada de refinamento para diferentes resoluções
        self.resolution_refine = nn.Sequential(
            nn.Conv2d(out_channels, f//2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f//2, out_channels, kernel_size=1),
            nn.Tanh()  # Para manter valores na faixa [-1, 1]
        )
    
    def _conv_block(self, in_ch, out_ch):
        """Bloco de convolução com BatchNorm e Dropout"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
    
    def forward(self, x, target_scale=None):
        # Encoder
        c1 = self.conv1(x)
        p1 = self.pool(c1)
        
        c2 = self.conv2(p1)
        p2 = self.pool(c2)
        
        c3 = self.conv3(p2)
        p3 = self.pool(c3)
        
        c4 = self.conv4(p3)
        p4 = self.pool(c4)
        
        # Bottleneck com atenção
        b_conv = self.bottleneck_conv(p4)
        attention = self.bottleneck(p4)
        b = b_conv * attention  # Aplicar atenção
        
        # Decoder
        u4 = self.up4(b)
        u4 = self._match_size_and_concat(u4, c4)
        u4 = self.conv_up4(u4)
        
        u3 = self.up3(u4)
        u3 = self._match_size_and_concat(u3, c3)
        u3 = self.conv_up3(u3)
        
        u2 = self.up2(u3)
        u2 = self._match_size_and_concat(u2, c2)
        u2 = self.conv_up2(u2)
        
        u1 = self.up1(u2)
        u1 = self._match_size_and_concat(u1, c1)
        u1 = self.conv_up1(u1)
        
        # Saída
        output = self.final_conv(u1)
        
        # Refinamento baseado na escala
        if target_scale is not None and target_scale != 1.0:
            refinement = self.resolution_refine(output)
            output = output + 0.1 * refinement  # Residual connection
        
        return output
    
    def _match_size_and_concat(self, upsampled, skip):
        """Ajusta tamanho e concatena skip connections"""
        if upsampled.size() != skip.size():
            upsampled = F.interpolate(upsampled, size=skip.size()[2:], 
                                    mode='bilinear', align_corners=False)
        return torch.cat([upsampled, skip], 1)


class PerceptualLoss(nn.Module):
    """Loss perceptual para melhor qualidade visual"""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
    
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        l1_loss = self.l1(pred, target)
        
        # Gradient loss para preservar detalhes
        grad_loss = self._gradient_loss(pred, target)
        
        return mse_loss + 0.1 * l1_loss + 0.01 * grad_loss
    
    def _gradient_loss(self, pred, target):
        """Calcula loss baseado em gradientes"""
        def gradient(img):
            grad_x = img[:, :, :, 1:] - img[:, :, :, :-1]
            grad_y = img[:, :, 1:, :] - img[:, :, :-1, :]
            return grad_x, grad_y
        
        pred_grad_x, pred_grad_y = gradient(pred)
        target_grad_x, target_grad_y = gradient(target)
        
        loss_x = F.l1_loss(pred_grad_x, target_grad_x)
        loss_y = F.l1_loss(pred_grad_y, target_grad_y)
        
        return loss_x + loss_y


def train_adaptive_unet(low_res_files, high_res_files, target_resolution=None, 
                       epochs=100, batch_size=4, learning_rate=1e-4, 
                       patch_size=128, save_path="adaptive_unet.pth"):
    """
    Treina a U-Net adaptativa
    """
    print(f"Iniciando treinamento para resolução alvo: {target_resolution}m")
    
    # Dataset e DataLoader
    dataset = AdaptiveSuperResTiffDataset(
        low_res_files, high_res_files, 
        target_resolution=target_resolution,
        patch_size=patch_size
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                           num_workers=0, pin_memory=True if torch.cuda.is_available() else False)
    
    # Modelo, otimizador e loss
    model = ResolutionAwareUNet(base_filters=32).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = PerceptualLoss()
    
    # Treinamento
    model.train()
    best_loss = float('inf')
    
    print("Iniciando treinamento...")
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, (low_res, high_res, resolution_factors) in enumerate(dataloader):
            low_res = low_res.to(device)
            high_res = high_res.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass com informação de escala
            avg_resolution_factor = resolution_factors.mean().item()
            predictions = model(low_res, target_scale=avg_resolution_factor)
            
            # Calcular loss
            loss = criterion(predictions, high_res)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Log progresso
            if batch_idx % 50 == 0:
                print(f"Época [{epoch+1}/{epochs}], Batch [{batch_idx}], "
                      f"Loss: {loss.item():.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Atualizar learning rate
        scheduler.step()
        
        # Média da loss da época
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
        print(f"Época [{epoch+1}/{epochs}] - Loss média: {avg_epoch_loss:.6f}")
        
        # Salvar melhor modelo
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'target_resolution': target_resolution,
            }, save_path)
            print(f"✅ Modelo salvo com loss: {best_loss:.6f}")
    
    print("Treinamento concluído!")
    return model


def generate_super_resolution(model_path, input_raster_path, output_path, 
                            target_resolution=None, device=None):
    """
    Gera super resolução de um raster usando modelo treinado
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Carregar modelo
    checkpoint = torch.load(model_path, map_location=device)
    model = ResolutionAwareUNet(base_filters=32).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Modelo carregado. Resolução de treinamento: {checkpoint.get('target_resolution', 'N/A')}m")
    
    # Processar raster
    with rasterio.open(input_raster_path) as src:
        # Ler dados
        img = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        
        # Obter resolução atual
        current_res_x = abs(src.transform.a)
        current_res_y = abs(src.transform.e)
        current_resolution = (current_res_x + current_res_y) / 2
        
        print(f"Resolução atual: {current_resolution:.2f}m")
        
        # Calcular fator de escala se resolução alvo especificada
        if target_resolution:
            scale_factor = current_resolution / target_resolution
            print(f"Fator de escala para {target_resolution}m: {scale_factor:.2f}x")
        else:
            scale_factor = 2.0  # Padrão: 2x
            target_resolution = current_resolution / scale_factor
        
        # Normalizar imagem
        dataset_dummy = AdaptiveSuperResTiffDataset([input_raster_path], [input_raster_path])
        img_norm = dataset_dummy._normalize_image(img, src.nodata)
        
        # Converter para tensor
        img_tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(device)
        
        # Inferência
        with torch.no_grad():
            if scale_factor != 1.0:
                # Redimensionar input se necessário
                if scale_factor > 1:
                    new_height = int(img.shape[0] * scale_factor)
                    new_width = int(img.shape[1] * scale_factor)
                    img_tensor = F.interpolate(img_tensor, size=(new_height, new_width), 
                                             mode='bilinear', align_corners=False)
            
            # Aplicar modelo
            output = model(img_tensor, target_scale=scale_factor)
            output_array = output.cpu().numpy().squeeze()
        
        # Desnormalizar (aproximação simples)
        valid_mask = img_norm > 0
        if np.any(valid_mask):
            original_min = np.min(img[valid_mask])
            original_max = np.max(img[valid_mask])
            output_array = output_array * (original_max - original_min) + original_min
        
        # Atualizar perfil para nova resolução
        if scale_factor != 1.0:
            profile.update({
                'height': output_array.shape[0],
                'width': output_array.shape[1],
                'transform': src.transform * src.transform.scale(1/scale_factor, 1/scale_factor)
            })
        
        # Salvar resultado
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(output_array.astype(profile['dtype']), 1)
        
        print(f"✅ Super resolução gerada: {output_path}")
        print(f"   Resolução final: {target_resolution:.2f}m")
        print(f"   Dimensões: {output_array.shape}")


# Exemplo de uso
if __name__ == "__main__":
    # Arquivos de exemplo - substitua pelos seus caminhos
    low_res_files = ["dados/rec_geosampa_10m.tif"]
    high_res_files = ["dados/rec_geosampa_30m.tif"]
    
    # Treinar modelo para resolução específica
    target_resolution = 10  
    model = train_adaptive_unet(
        low_res_files, high_res_files, 
        target_resolution=target_resolution,
        epochs=120,
        batch_size=2,
        patch_size=128
    )
    
    # Gerar super resolução
    generate_super_resolution(
        "adaptive_unet.pth",
        "dados/rec_anadem_teste.tif",
        "output/anadem_teste_16f_80ep_10m.tif",
        target_resolution=10
    )

        # Gerar super resolução
    generate_super_resolution(
        "adaptive_unet.pth",
        "dados/rec_anadem.tif",
        "output/anadem_16f_80ep_10m.tif",
        target_resolution=10
    )