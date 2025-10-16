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
        self.target_resolution = target_resolution
        
        assert len(self.low_res_datasets) == len(self.high_res_datasets), \
            "Número de arquivos low-res e high-res deve ser igual"
        
        self._calculate_resolution_info()
    
    def _calculate_resolution_info(self):
        """Calcula informações de resolução dos dados"""
        self.resolution_info = []
        
        for low_src, high_src in zip(self.low_res_datasets, self.high_res_datasets):
            low_res_x = abs(low_src.transform.a)
            low_res_y = abs(low_src.transform.e)
            
            high_res_x = abs(high_src.transform.a)
            high_res_y = abs(high_src.transform.e)
            
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
    
    def __len__(self):
        return 10000
    
    def __getitem__(self, idx):
        file_idx = np.random.randint(0, len(self.low_res_datasets))
        low_src = self.low_res_datasets[file_idx]
        high_src = self.high_res_datasets[file_idx]
        
        scale_factor = self.resolution_info[file_idx]['avg_scale']
        
        # Calcular tamanho do patch high-res
        high_patch_size = int(self.patch_size * scale_factor)
        
        max_x = high_src.width - high_patch_size
        max_y = high_src.height - high_patch_size
        
        if max_x <= 0 or max_y <= 0:
            window_high = Window(0, 0, min(high_src.width, high_patch_size), 
                                min(high_src.height, high_patch_size))
            window_low = Window(0, 0, min(low_src.width, self.patch_size),
                               min(low_src.height, self.patch_size))
        else:
            random_x = np.random.randint(0, max_x)
            random_y = np.random.randint(0, max_y)
            window_high = Window(random_x, random_y, high_patch_size, high_patch_size)
            
            low_x = int(random_x / scale_factor)
            low_y = int(random_y / scale_factor)
            window_low = Window(low_x, low_y, self.patch_size, self.patch_size)
        
        try:
            img_low = low_src.read(1, window=window_low).astype(np.float32)
            img_high = high_src.read(1, window=window_high).astype(np.float32)
        except Exception as e:
            print(f"Erro ao ler patch: {e}")
            img_low = low_src.read(1).astype(np.float32)[:self.patch_size, :self.patch_size]
            img_high = high_src.read(1).astype(np.float32)[:high_patch_size, :high_patch_size]
        
        # Normalização robusta usando percentis
        img_low_norm = self._normalize_image(img_low, low_src.nodata)
        img_high_norm = self._normalize_image(img_high, high_src.nodata)
        
        # Garantir que high-res seja exatamente scale_factor vezes maior
        target_h = int(img_low_norm.shape[0] * scale_factor)
        target_w = int(img_low_norm.shape[1] * scale_factor)
        
        if img_high_norm.shape != (target_h, target_w):
            img_high_norm_tensor = torch.from_numpy(img_high_norm).unsqueeze(0).unsqueeze(0)
            img_high_norm_tensor = F.interpolate(img_high_norm_tensor, size=(target_h, target_w),
                                                mode='bilinear', align_corners=False)
            img_high_norm = img_high_norm_tensor.squeeze().numpy()
        
        low_tensor = torch.from_numpy(img_low_norm).unsqueeze(0)
        high_tensor = torch.from_numpy(img_high_norm).unsqueeze(0)
        
        if self.transform:
            low_tensor = self.transform(low_tensor)
            high_tensor = self.transform(high_tensor)
        
        resolution_factor = scale_factor
        
        return low_tensor, high_tensor, resolution_factor
    
    def _normalize_image(self, img, nodata_value):
        """Normaliza imagem de forma robusta e generalizada"""
        if nodata_value is not None:
            valid_mask = img != nodata_value
        else:
            valid_mask = np.isfinite(img)
        
        if not np.any(valid_mask):
            # Retorna array de zeros do mesmo shape
            return np.zeros_like(img, dtype=np.float32)
        
        valid_data = img[valid_mask]
        
        # Usar percentis para normalização mais robusta
        p2, p98 = np.percentile(valid_data, [2, 98])
        
        if p98 - p2 > 0:
            img_norm = np.zeros_like(img)
            img_norm[valid_mask] = np.clip((img[valid_mask] - p2) / (p98 - p2), 0, 1)
            
            # Data augmentation: adicionar ruído aleatório para generalização
            if np.random.random() > 0.5:
                noise = np.random.normal(0, 0.02, img_norm.shape).astype(np.float32)
                img_norm = np.clip(img_norm + noise, 0, 1)
            
            return img_norm
        else:
            # Se não há variação, retorna zeros
            return np.zeros_like(img, dtype=np.float32)
    
    def __del__(self):
        for dataset in self.low_res_datasets + self.high_res_datasets:
            if hasattr(dataset, 'close'):
                dataset.close()


class ResolutionAwareUNet(nn.Module):
    """
    U-Net melhorada com sub-pixel convolution para super-resolução
    """
    def __init__(self, in_channels=1, out_channels=1, base_filters=32, scale_factor=6):
        super().__init__()
        
        self.scale_factor = scale_factor
        f = base_filters
        
        # Encoder com mais camadas para 30m->5m (6x)
        self.conv1 = self._conv_block(in_channels, f)
        self.conv2 = self._conv_block(f, f*2)
        self.conv3 = self._conv_block(f*2, f*4)
        self.conv4 = self._conv_block(f*4, f*8)
        self.conv5 = self._conv_block(f*8, f*16)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = self._conv_block(f*16, f*32)
        
        # Decoder
        self.up5 = nn.ConvTranspose2d(f*32, f*16, kernel_size=2, stride=2)
        self.conv_up5 = self._conv_block(f*32, f*16)
        
        self.up4 = nn.ConvTranspose2d(f*16, f*8, kernel_size=2, stride=2)
        self.conv_up4 = self._conv_block(f*16, f*8)
        
        self.up3 = nn.ConvTranspose2d(f*8, f*4, kernel_size=2, stride=2)
        self.conv_up3 = self._conv_block(f*8, f*4)
        
        self.up2 = nn.ConvTranspose2d(f*4, f*2, kernel_size=2, stride=2)
        self.conv_up2 = self._conv_block(f*4, f*2)
        
        self.up1 = nn.ConvTranspose2d(f*2, f, kernel_size=2, stride=2)
        self.conv_up1 = self._conv_block(f*2, f)
        
        # Sub-pixel convolution para upsampling final
        self.subpixel_conv = nn.Sequential(
            nn.Conv2d(f, f*4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),  # 2x upsampling
            nn.ReLU(inplace=True),
            nn.Conv2d(f, f*4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),  # Mais 2x = 4x total
            nn.ReLU(inplace=True)
        )
        
        # Camada final de refinamento
        self.final_refine = nn.Sequential(
            nn.Conv2d(f, f//2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f//2, out_channels, kernel_size=1)
        )
    
    def _conv_block(self, in_ch, out_ch):
        """Bloco de convolução melhorado"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.05)
        )
    
    def forward(self, x, target_scale=None):
        # Guardar tamanho original
        _, _, h, w = x.shape
        
        # Encoder
        c1 = self.conv1(x)
        p1 = self.pool(c1)
        
        c2 = self.conv2(p1)
        p2 = self.pool(c2)
        
        c3 = self.conv3(p2)
        p3 = self.pool(c3)
        
        c4 = self.conv4(p3)
        p4 = self.pool(c4)
        
        c5 = self.conv5(p4)
        p5 = self.pool(c5)
        
        # Bottleneck
        b = self.bottleneck(p5)
        
        # Decoder
        u5 = self.up5(b)
        u5 = self._match_size_and_concat(u5, c5)
        u5 = self.conv_up5(u5)
        
        u4 = self.up4(u5)
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
        
        # Sub-pixel convolution para 4x upsampling
        u_upsampled = self.subpixel_conv(u1)
        
        # Ajuste final para 6x (interpolar de 4x para 6x)
        if target_scale is not None:
            target_h = int(h * target_scale)
            target_w = int(w * target_scale)
            u_upsampled = F.interpolate(u_upsampled, size=(target_h, target_w),
                                       mode='bilinear', align_corners=False)
        
        # Refinamento final
        output = self.final_refine(u_upsampled)
        
        return output
    
    def _match_size_and_concat(self, upsampled, skip):
        """Ajusta tamanho e concatena skip connections"""
        if upsampled.size() != skip.size():
            upsampled = F.interpolate(upsampled, size=skip.size()[2:], 
                                    mode='bilinear', align_corners=False)
        return torch.cat([upsampled, skip], 1)


class PerceptualLoss(nn.Module):
    """Loss combinada para melhor qualidade"""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
    
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        l1_loss = self.l1(pred, target)
        grad_loss = self._gradient_loss(pred, target)
        ssim_loss = self._ssim_loss(pred, target)
        
        # Pesos balanceados - SSIM ajuda com textura
        return 0.4 * mse_loss + 0.2 * l1_loss + 0.2 * grad_loss + 0.2 * (1 - ssim_loss)
    
    def _gradient_loss(self, pred, target):
        """Preserva detalhes finos"""
        def gradient(img):
            grad_x = img[:, :, :, 1:] - img[:, :, :, :-1]
            grad_y = img[:, :, 1:, :] - img[:, :, :-1, :]
            return grad_x, grad_y
        
        pred_grad_x, pred_grad_y = gradient(pred)
        target_grad_x, target_grad_y = gradient(target)
        
        loss_x = F.l1_loss(pred_grad_x, target_grad_x)
        loss_y = F.l1_loss(pred_grad_y, target_grad_y)
        
        return loss_x + loss_y
    
    def _ssim_loss(self, pred, target, window_size=11):
        """SSIM loss para preservar estrutura e textura"""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu_pred = F.avg_pool2d(pred, window_size, stride=1, padding=window_size//2)
        mu_target = F.avg_pool2d(target, window_size, stride=1, padding=window_size//2)
        
        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target
        
        sigma_pred_sq = F.avg_pool2d(pred ** 2, window_size, stride=1, padding=window_size//2) - mu_pred_sq
        sigma_target_sq = F.avg_pool2d(target ** 2, window_size, stride=1, padding=window_size//2) - mu_target_sq
        sigma_pred_target = F.avg_pool2d(pred * target, window_size, stride=1, padding=window_size//2) - mu_pred_target
        
        ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
                   ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))
        
        return ssim_map.mean()


def train_adaptive_unet(low_res_files, high_res_files, target_resolution=None, 
                       epochs=100, batch_size=4, learning_rate=1e-4, 
                       patch_size=128, save_path="model/adaptive_unet.pth"):
    """
    Treina a U-Net adaptativa
    """
    print(f"Iniciando treinamento para resolução alvo: {target_resolution}m")
    start_time = time.time()
    
    dataset = AdaptiveSuperResTiffDataset(
        low_res_files, high_res_files, 
        target_resolution=target_resolution,
        patch_size=patch_size
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                           num_workers=0, pin_memory=True if torch.cuda.is_available() else False)
    
    # Calcular scale_factor para o modelo
    scale_factor = dataset.resolution_info[0]['avg_scale']
    
    model = ResolutionAwareUNet(base_filters=32, scale_factor=int(scale_factor)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                     patience=10, verbose=False)
    criterion = PerceptualLoss().to(device)
    
    model.train()
    best_loss = float('inf')
    
    print(f"Treinando com scale_factor={scale_factor}")
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, (low_res, high_res, resolution_factors) in enumerate(dataloader):
            low_res = low_res.to(device)
            high_res = high_res.to(device)
            
            optimizer.zero_grad()
            
            avg_resolution_factor = resolution_factors.mean().item()
            predictions = model(low_res, target_scale=avg_resolution_factor)
            
            # Garantir mesmas dimensões
            if predictions.shape != high_res.shape:
                predictions = F.interpolate(predictions, size=high_res.shape[2:],
                                           mode='bilinear', align_corners=False)
            
            loss = criterion(predictions, high_res)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 50 == 0:
                print(f"Época [{epoch+1}/{epochs}], Batch [{batch_idx}], "
                      f"Loss: {loss.item():.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
        print(f"Época [{epoch+1}/{epochs}] - Loss média: {avg_epoch_loss:.6f}")
        
        scheduler.step(avg_epoch_loss)
        
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'target_resolution': target_resolution,
                'scale_factor': scale_factor,
            }, save_path)
            print(f"✅ Modelo salvo com loss: {best_loss:.6f}")
    
    elapsed = time.time() - start_time
    print(f"Tempo total de treinamento: {elapsed/60:.2f} minutos")
    return model


def generate_super_resolution(
    model_path, input_raster_path, output_path,
    target_resolution=None, device=None, tile_size=256, overlap=0.5
):
    """
    Gera super-resolução com blending suave usando janelas Hann
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device)
    scale_factor = checkpoint.get('scale_factor', 6)
    
    model = ResolutionAwareUNet(base_filters=32, scale_factor=int(scale_factor)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Modelo carregado - Scale factor: {scale_factor}x")

    with rasterio.open(input_raster_path) as src:
        img = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata
        current_res_x = abs(src.transform.a)
        current_res_y = abs(src.transform.e)
        current_resolution = (current_res_x + current_res_y) / 2

        print(f"Dimensões entrada: {img.shape}")
        print(f"Resolução atual: {current_resolution:.2f}m")

        if target_resolution:
            calc_scale = current_resolution / target_resolution
            print(f"Resolução alvo: {target_resolution}m")
            print(f"Scale calculado: {calc_scale:.2f}x")
        else:
            calc_scale = scale_factor
            target_resolution = current_resolution / calc_scale

        # Normalização global robusta
        valid_mask = np.isfinite(img)
        if nodata is not None:
            valid_mask &= (img != nodata)
        
        if not np.any(valid_mask):
            print("ERRO: Imagem sem dados válidos!")
            return
        
        valid_data = img[valid_mask]
        p2, p98 = np.percentile(valid_data, [2, 98])
        
        print(f"Percentis globais: p2={p2:.2f}, p98={p98:.2f}")
        print(f"Range entrada: [{valid_data.min():.2f}, {valid_data.max():.2f}]")
        
        img_norm = np.zeros_like(img)
        img_norm[valid_mask] = np.clip((img[valid_mask] - p2) / (p98 - p2 + 1e-8), 0, 1)

        # Dimensões de saída corretas
        out_height = int(np.ceil(img.shape[0] * calc_scale))
        out_width = int(np.ceil(img.shape[1] * calc_scale))
        
        print(f"Dimensões saída: ({out_height}, {out_width})")
        
        output_array = np.zeros((out_height, out_width), dtype=np.float32)
        weight_array = np.zeros_like(output_array)

        # Criar janela Hann 2D para blending suave
        hann_window = np.outer(np.hanning(tile_size), np.hanning(tile_size))

        step = int(tile_size * (1 - overlap))
        
        # Calcular número de tiles necessários para cobrir a imagem inteira
        n_tiles_h = int(np.ceil((img.shape[0] - tile_size) / step)) + 1 if img.shape[0] > tile_size else 1
        n_tiles_w = int(np.ceil((img.shape[1] - tile_size) / step)) + 1 if img.shape[1] > tile_size else 1
        total_tiles = n_tiles_h * n_tiles_w
        
        processed = 0

        print(f"Processando {total_tiles} tiles ({n_tiles_h}x{n_tiles_w}) com overlap={overlap}")
        print(f"Tile size: {tile_size}, Step: {step}")
        
        for i in range(0, img.shape[0], step):
            for j in range(0, img.shape[1], step):
                # Extrair tile
                i_end = min(i + tile_size, img.shape[0])
                j_end = min(j + tile_size, img.shape[1])
                
                tile = img_norm[i:i_end, j:j_end]
                tile_h, tile_w = tile.shape
                
                # Se tile é menor que tile_size, fazer padding
                tile_padded = np.zeros((tile_size, tile_size), dtype=np.float32)
                tile_padded[:tile_h, :tile_w] = tile
                
                # Criar máscara de peso baseada no tamanho real do tile
                weight_mask = np.zeros((tile_size, tile_size), dtype=np.float32)
                weight_mask[:tile_h, :tile_w] = hann_window[:tile_h, :tile_w]
                
                # Processar tile
                tile_tensor = torch.from_numpy(tile_padded).unsqueeze(0).unsqueeze(0).to(device).float()
                
                with torch.no_grad():
                    tile_out = model(tile_tensor, target_scale=calc_scale)
                    tile_out_np = tile_out.cpu().numpy().squeeze()
                
                # Dimensões de saída do tile
                tile_out_h = int(np.ceil(tile_h * calc_scale))
                tile_out_w = int(np.ceil(tile_w * calc_scale))
                
                # Extrair apenas a parte válida (sem padding)
                tile_out_full_h = int(np.ceil(tile_size * calc_scale))
                tile_out_full_w = int(np.ceil(tile_size * calc_scale))
                
                # Redimensionar se necessário
                if tile_out_np.shape != (tile_out_full_h, tile_out_full_w):
                    tile_out_tensor = torch.from_numpy(tile_out_np).unsqueeze(0).unsqueeze(0)
                    tile_out_tensor = F.interpolate(
                        tile_out_tensor, 
                        size=(tile_out_full_h, tile_out_full_w),
                        mode='bilinear', 
                        align_corners=False
                    )
                    tile_out_np = tile_out_tensor.squeeze().cpu().numpy()
                
                # Extrair apenas região válida (correspondente ao tile original sem padding)
                tile_out_valid = tile_out_np[:tile_out_h, :tile_out_w]
                
                # Upsample dos pesos
                weights_tensor = torch.from_numpy(weight_mask).unsqueeze(0).unsqueeze(0)
                weights_upsampled = F.interpolate(
                    weights_tensor, 
                    size=(tile_out_full_h, tile_out_full_w),
                    mode='bilinear', 
                    align_corners=False
                )
                weights_valid = weights_upsampled.squeeze().cpu().numpy()[:tile_out_h, :tile_out_w]
                
                # Calcular posição na imagem de saída
                out_i = int(np.round(i * calc_scale))
                out_j = int(np.round(j * calc_scale))
                
                # Garantir que não ultrapasse os limites
                out_i_end = min(out_i + tile_out_h, out_height)
                out_j_end = min(out_j + tile_out_w, out_width)
                
                actual_h = out_i_end - out_i
                actual_w = out_j_end - out_j
                
                # Acumular com pesos
                output_array[out_i:out_i_end, out_j:out_j_end] += tile_out_valid[:actual_h, :actual_w] * weights_valid[:actual_h, :actual_w]
                weight_array[out_i:out_i_end, out_j:out_j_end] += weights_valid[:actual_h, :actual_w]
                
                processed += 1
                if processed % 10 == 0:
                    print(f"Progresso: {processed}/{total_tiles} tiles - Posição: ({i}, {j}) -> ({out_i}, {out_j})")

        print(f"Total processado: {processed} tiles")

        # Normalizar por pesos
        valid_weights = weight_array > 1e-6
        print(f"Pixels com peso válido: {valid_weights.sum()} / {output_array.size}")
        
        if not np.any(valid_weights):
            print("ERRO: Nenhum pixel com peso válido!")
            return
        
        output_array[valid_weights] /= weight_array[valid_weights]

        # Desnormalizar
        output_array = output_array * (p98 - p2 + 1e-8) + p2

        print(f"Output range final: [{output_array[valid_weights].min():.2f}, {output_array[valid_weights].max():.2f}]")

        # Atualizar profile
        new_transform = src.transform * src.transform.scale(
            (1.0 / calc_scale),
            (1.0 / calc_scale)
        )
        
        profile.update({
            'height': out_height,
            'width': out_width,
            'transform': new_transform,
            'dtype': 'float32'
        })
        
        if nodata is not None:
            # Marcar áreas sem peso como nodata
            output_array[~valid_weights] = float(nodata)

        # Salvar
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(output_array.astype('float32'), 1)

        print(f"✅ Super resolução gerada: {output_path}")
        print(f"   Resolução final: {target_resolution:.2f}m")
        print(f"   Dimensões finais: {output_array.shape}")
        print(f"   Transform: {new_transform}")


def visualize_comparison(original_path, output_path, sample_size=512):
    """Visualiza comparação lado a lado"""
    with rasterio.open(original_path) as src1:
        orig = src1.read(1, window=Window(0, 0, sample_size, sample_size))
    
    with rasterio.open(output_path) as src2:
        out = src2.read(1, window=Window(0, 0, sample_size*6, sample_size*6))
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    axes[0].imshow(orig, cmap='viridis')
    axes[0].set_title('Original 30m')
    axes[0].axis('off')
    
    axes[1].imshow(out, cmap='viridis')
    axes[1].set_title('Super-res 5m')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('comparison.png', dpi=150, bbox_inches='tight')
    print("✅ Comparação salva em comparison.png")


# Exemplo de uso
if __name__ == "__main__":
    low_res_files = ["dados/geosampa_30m.tif"]
    high_res_files = ["dados/geosampa_5m.tif"]
    
    # Treinar modelo
    model = train_adaptive_unet(
        low_res_files, high_res_files, 
        target_resolution=5,
        epochs=80,             
        batch_size=2,
        patch_size=128,
        save_path="model/adaptive_unet_5m_v2.pth"
    )
    
    # Gerar super resolução
    generate_super_resolution(
        "model/adaptive_unet_5m_v2.pth",
        "dados/ANADEM_Recorte_IPT.tif",
        "output/ANADEM_Recorte_IPT_80ep_improved.tif",
        target_resolution=5,
        tile_size=256,
        overlap=0.5
    )