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


class DEMRefinementDataset(Dataset):
    """
    Dataset para refinamento de DEM: duas entradas de mesma resolução (5m)
    - input1: ANADEM (menos detalhado, 5m)
    - input2: GeoSampa (mais detalhado, 5m) 
    - target: GeoSampa (ground truth para treinamento)
    """
    def __init__(self, input1_files, input2_files, target_files=None, transform=None, patch_size=128):
        self.input1_datasets = [rasterio.open(f) for f in input1_files]
        self.input2_datasets = [rasterio.open(f) for f in input2_files]
        # Se target_files não for fornecido, usa input2 como target
        self.target_datasets = [rasterio.open(f) for f in (target_files or input2_files)]
        
        self.transform = transform
        self.patch_size = patch_size
        
        assert len(self.input1_datasets) == len(self.input2_datasets) == len(self.target_datasets), \
            "Número de arquivos deve ser igual para todas as entradas"
        
        self._check_resolutions()
    
    def _check_resolutions(self):
        """Verifica se todas as resoluções são compatíveis"""
        for idx, (src1, src2, tgt) in enumerate(zip(self.input1_datasets, 
                                                      self.input2_datasets, 
                                                      self.target_datasets)):
            res1_x = abs(src1.transform.a)
            res2_x = abs(src2.transform.a)
            tgt_x = abs(tgt.transform.a)
            
            print(f"Arquivo {idx+1}: Input1={res1_x:.2f}m, Input2={res2_x:.2f}m, Target={tgt_x:.2f}m")
    
    def __len__(self):
        return 10000  # Número de patches por época
    
    def __getitem__(self, idx):
        file_idx = np.random.randint(0, len(self.input1_datasets))
        src1 = self.input1_datasets[file_idx]
        src2 = self.input2_datasets[file_idx]
        tgt = self.target_datasets[file_idx]
        
        # Selecionar região aleatória
        max_x = min(src1.width, src2.width, tgt.width) - self.patch_size
        max_y = min(src1.height, src2.height, tgt.height) - self.patch_size
        
        if max_x <= 0 or max_y <= 0:
            window = Window(0, 0, self.patch_size, self.patch_size)
        else:
            random_x = np.random.randint(0, max_x)
            random_y = np.random.randint(0, max_y)
            window = Window(random_x, random_y, self.patch_size, self.patch_size)
        
        try:
            img1 = src1.read(1, window=window).astype(np.float32)
            img2 = src2.read(1, window=window).astype(np.float32)
            img_target = tgt.read(1, window=window).astype(np.float32)
        except Exception as e:
            print(f"Erro ao ler patch: {e}")
            img1 = src1.read(1).astype(np.float32)[:self.patch_size, :self.patch_size]
            img2 = src2.read(1).astype(np.float32)[:self.patch_size, :self.patch_size]
            img_target = tgt.read(1).astype(np.float32)[:self.patch_size, :self.patch_size]
        
        # Normalização
        img1_norm = self._normalize_image(img1, src1.nodata)
        img2_norm = self._normalize_image(img2, src2.nodata)
        img_target_norm = self._normalize_image(img_target, tgt.nodata)
        
        # Concatenar as duas entradas em canais diferentes
        input_tensor = torch.stack([
            torch.from_numpy(img1_norm),
            torch.from_numpy(img2_norm)
        ])  # Shape: [2, H, W]
        
        target_tensor = torch.from_numpy(img_target_norm).unsqueeze(0)  # Shape: [1, H, W]
        
        if self.transform:
            input_tensor = self.transform(input_tensor)
            target_tensor = self.transform(target_tensor)
        
        return input_tensor, target_tensor
    
    def _normalize_image(self, img, nodata_value):
        """Normaliza imagem de forma robusta"""
        if nodata_value is not None:
            valid_mask = img != nodata_value
        else:
            valid_mask = np.isfinite(img)
        
        if not np.any(valid_mask):
            return np.zeros_like(img, dtype=np.float32)
        
        valid_data = img[valid_mask]
        p2, p98 = np.percentile(valid_data, [2, 98])
        
        if p98 - p2 > 0:
            img_norm = np.zeros_like(img)
            img_norm[valid_mask] = np.clip((img[valid_mask] - p2) / (p98 - p2), 0, 1)
            
            # Data augmentation leve
            if np.random.random() > 0.5:
                noise = np.random.normal(0, 0.02, img_norm.shape).astype(np.float32)
                img_norm = np.clip(img_norm + noise, 0, 1)
            
            return img_norm
        else:
            return np.zeros_like(img, dtype=np.float32)
    
    def __del__(self):
        for dataset in (self.input1_datasets + self.input2_datasets + self.target_datasets):
            if hasattr(dataset, 'close'):
                dataset.close()


class ConvBlock(nn.Module):
    """
    Bloco de convolução seguindo a referência:
    Conv -> ReLU -> Conv -> ReLU (com dropout opcional)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, use_dropout=False, dropout_rate=0.1):
        super(ConvBlock, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, 
                               padding=padding, padding_mode='replicate')
        self.activation1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 
                               padding=padding, padding_mode='replicate')
        self.activation2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate) if use_dropout else nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.activation2(x)
        return x


class DEMRefinementUNet(nn.Module):
    """
    U-Net para refinamento de DEM com mesma resolução entrada/saída
    Entrada: 2 canais (ANADEM + GeoSampa, ambos 5m)
    Saída: 1 canal (DEM refinado, 5m)
    """
    def __init__(self, in_channels=2, out_channels=1, base_filters=32, depth=4, 
                 kernel_size=3, pool_size=2, dropout_rate=0.1):
        super(DEMRefinementUNet, self).__init__()
        
        self.depth = depth
        
        # Contracting Path (Encoder) - 4 níveis
        self.down_blocks = nn.ModuleList()
        for i in range(depth):
            in_ch = in_channels if i == 0 else base_filters * (2 ** (i - 1))
            out_ch = base_filters * (2 ** i)
            use_dropout = True  # Dropout em todos os níveis do encoder
            self.down_blocks.append(ConvBlock(in_ch, out_ch, kernel_size, use_dropout, dropout_rate))
        
        self.pool = nn.MaxPool2d(pool_size)
        
        # Bottleneck
        bottleneck_channels = base_filters * (2 ** (depth - 1))
        self.bottleneck = ConvBlock(bottleneck_channels, bottleneck_channels * 2, 
                                   kernel_size, use_dropout=True, dropout_rate=dropout_rate)
        
        # Expanding Path (Decoder) - 4 níveis (mesma quantidade do encoder)
        self.up_transpose_blocks = nn.ModuleList()
        self.up_conv_blocks = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            in_ch = bottleneck_channels * 2 if i == depth - 1 else base_filters * (2 ** (i + 1))
            out_ch = base_filters * (2 ** i)
            self.up_transpose_blocks.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2))
            # Skip connection dobra os canais
            self.up_conv_blocks.append(ConvBlock(out_ch * 2, out_ch, kernel_size, 
                                                use_dropout=False, dropout_rate=dropout_rate))
        
        # Final Convolution (1x1)
        self.final = nn.Conv2d(base_filters, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Contracting Path
        skip_connections = []
        for down_block in self.down_blocks:
            x = down_block(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Expanding Path
        skip_connections = list(reversed(skip_connections))
        for up_transpose, up_conv, skip_connection in zip(self.up_transpose_blocks, 
                                                           self.up_conv_blocks, 
                                                           skip_connections):
            x = up_transpose(x)
            # Ajustar tamanho se necessário
            if x.shape[2:] != skip_connection.shape[2:]:
                x = F.interpolate(x, size=skip_connection.shape[2:], 
                                mode='bilinear', align_corners=False)
            # Concatenar skip connection
            x = torch.cat((x, skip_connection), dim=1)
            x = up_conv(x)
        
        # Final Convolution
        x = self.final(x)
        return x


class PerceptualLoss(nn.Module):
    """Loss combinada para preservar detalhes e estrutura"""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
    
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        l1_loss = self.l1(pred, target)
        grad_loss = self._gradient_loss(pred, target)
        
        # Pesos balanceados
        return 0.4 * mse_loss + 0.3 * l1_loss + 0.3 * grad_loss
    
    def _gradient_loss(self, pred, target):
        """Preserva detalhes finos (importante para topografia)"""
        def gradient(img):
            grad_x = img[:, :, :, 1:] - img[:, :, :, :-1]
            grad_y = img[:, :, 1:, :] - img[:, :, :-1, :]
            return grad_x, grad_y
        
        pred_grad_x, pred_grad_y = gradient(pred)
        target_grad_x, target_grad_y = gradient(target)
        
        loss_x = F.l1_loss(pred_grad_x, target_grad_x)
        loss_y = F.l1_loss(pred_grad_y, target_grad_y)
        
        return loss_x + loss_y


def train_dem_refinement(input1_files, input2_files, target_files=None,
                        epochs=100, batch_size=4, learning_rate=1e-4, 
                        patch_size=128, save_path="model/dem_refinement_unet.pth"):
    """
    Treina a U-Net para refinamento de DEM
    """
    print("Iniciando treinamento de refinamento de DEM")
    start_time = time.time()
    
    dataset = DEMRefinementDataset(
        input1_files, input2_files, target_files,
        patch_size=patch_size
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                           num_workers=0, pin_memory=True if torch.cuda.is_available() else False)
    
    model = DEMRefinementUNet(
        in_channels=2, 
        out_channels=1, 
        base_filters=32, 
        depth=4,
        dropout_rate=0.1
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                     patience=10, verbose=True)
    criterion = PerceptualLoss().to(device)
    
    model.train()
    best_loss = float('inf')
    
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            
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
            }, save_path)
            print(f"✅ Modelo salvo com loss: {best_loss:.6f}")
    
    elapsed = time.time() - start_time
    print(f"Tempo total de treinamento: {elapsed/60:.2f} minutos")
    return model


def generate_refined_dem(
    model_path, input1_path, input2_path=None, output_path=None,
    device=None, tile_size=256, overlap=0.5
):
    """
    Gera DEM refinado usando dois arquivos de entrada (mesma resolução).
    Se input2_path for None, usa input1_path como segundo canal (útil quando não há GeoSampa).
    Se output_path for None, salva em "output/anadem_u_net.tif".
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if output_path is None:
        output_path = "output/anadem_u_net.tif"

    checkpoint = torch.load(model_path, map_location=device)
    
    model = DEMRefinementUNet(in_channels=2, out_channels=1, base_filters=32, depth=4).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("Modelo carregado para refinamento de DEM")

    # Se input2_path não for fornecido, usa input1_path como fallback
    if input2_path is None:
        print("Aviso: input2_path não fornecido. Usando input1_path como segundo canal.")
        input2_path = input1_path

    with rasterio.open(input1_path) as src1, rasterio.open(input2_path) as src2:
        img1 = src1.read(1).astype(np.float32)
        img2 = src2.read(1).astype(np.float32)
        
        profile = src1.profile.copy()
        nodata1 = src1.nodata
        nodata2 = src2.nodata

        print(f"Dimensões input1: {img1.shape}")
        print(f"Dimensões input2: {img2.shape}")
        
        # Garantir que ambas têm o mesmo tamanho
        min_h = min(img1.shape[0], img2.shape[0])
        min_w = min(img1.shape[1], img2.shape[1])
        img1 = img1[:min_h, :min_w]
        img2 = img2[:min_h, :min_w]

        # Normalização global
        def normalize_global(img, nodata):
            valid_mask = np.isfinite(img)
            if nodata is not None:
                valid_mask &= (img != nodata)
            
            if not np.any(valid_mask):
                return np.zeros_like(img), (0.0, 1.0)
            
            valid_data = img[valid_mask]
            p2, p98 = np.percentile(valid_data, [2, 98])
            
            img_norm = np.zeros_like(img)
            img_norm[valid_mask] = np.clip((img[valid_mask] - p2) / (p98 - p2 + 1e-8), 0, 1)
            return img_norm, (p2, p98)
        
        img1_norm, stats1 = normalize_global(img1, nodata1)
        img2_norm, stats2 = normalize_global(img2, nodata2)
        
        print(f"Stats input1: p2={stats1[0]:.2f}, p98={stats1[1]:.2f}")
        print(f"Stats input2: p2={stats2[0]:.2f}, p98={stats2[1]:.2f}")

        out_height, out_width = img1.shape
        output_array = np.zeros((out_height, out_width), dtype=np.float32)
        weight_array = np.zeros_like(output_array)

        # Janela Hann para blending
        hann_window = np.outer(np.hanning(tile_size), np.hanning(tile_size))

        step = int(tile_size * (1 - overlap))
        n_tiles_h = int(np.ceil((out_height - tile_size) / step)) + 1 if out_height > tile_size else 1
        n_tiles_w = int(np.ceil((out_width - tile_size) / step)) + 1 if out_width > tile_size else 1
        total_tiles = n_tiles_h * n_tiles_w
        
        processed = 0
        print(f"Processando {total_tiles} tiles ({n_tiles_h}x{n_tiles_w})")
        
        for i in range(0, out_height, step):
            for j in range(0, out_width, step):
                i_end = min(i + tile_size, out_height)
                j_end = min(j + tile_size, out_width)
                
                tile1 = img1_norm[i:i_end, j:j_end]
                tile2 = img2_norm[i:i_end, j:j_end]
                # Garante menor shape comum entre os dois tiles (prevenção extra)
                th = min(tile1.shape[0], tile2.shape[0])
                tw = min(tile1.shape[1], tile2.shape[1])
                tile1 = tile1[:th, :tw]
                tile2 = tile2[:th, :tw]
                
                tile_h, tile_w = tile1.shape
                
                # Padding se necessário
                tile1_padded = np.zeros((tile_size, tile_size), dtype=np.float32)
                tile2_padded = np.zeros((tile_size, tile_size), dtype=np.float32)
                tile1_padded[:tile_h, :tile_w] = tile1
                tile2_padded[:tile_h, :tile_w] = tile2
                
                weight_mask = np.zeros((tile_size, tile_size), dtype=np.float32)
                weight_mask[:tile_h, :tile_w] = hann_window[:tile_h, :tile_w]
                
                # Criar tensor de entrada com 2 canais
                tile_input = torch.stack([
                    torch.from_numpy(tile1_padded),
                    torch.from_numpy(tile2_padded)
                ]).unsqueeze(0).to(device).float()  # Shape: [1, 2, H, W]
                
                with torch.no_grad():
                    tile_out = model(tile_input)
                    tile_out_np = tile_out.cpu().numpy().squeeze()
                
                # Extrair parte válida
                tile_out_valid = tile_out_np[:tile_h, :tile_w]
                weights_valid = weight_mask[:tile_h, :tile_w]
                
                # Acumular
                output_array[i:i+tile_h, j:j+tile_w] += tile_out_valid * weights_valid
                weight_array[i:i+tile_h, j:j+tile_w] += weights_valid
                
                processed += 1
                if processed % 10 == 0:
                    print(f"Progresso: {processed}/{total_tiles} tiles")

        # Normalizar por pesos
        valid_weights = weight_array > 1e-6
        if np.any(valid_weights):
            output_array[valid_weights] /= weight_array[valid_weights]
        
        # Desnormalizar usando stats do input2 (target)
        output_array = output_array * (stats2[1] - stats2[0] + 1e-8) + stats2[0]

        # Marcar nodata
        if nodata1 is not None:
            output_array[~valid_weights] = float(nodata1)

        # Salvar
        profile.update({'dtype': 'float32'})
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(output_array.astype('float32'), 1)

        print(f"✅ DEM refinado gerado: {output_path}")


# Exemplo de uso
if __name__ == "__main__":
    # Caminhos dos arquivos
    input1_files = ["dados/anadem_5m.tif"]  
    input2_files = ["dados/geosampa_5m_reprojetado.tif"] 
    # target_files usa input2_files por padrão (GeoSampa como ground truth)

    # # Treinar modelo
    # model = train_dem_refinement(
    #     input1_files, input2_files,
    #     epochs=10,
    #     batch_size=4,
    #     patch_size=128,
    #     save_path="model/dem_refinement_unet.pth"
    # )
    
    # Gerar DEM refinado em uma nova área (usando ANADEM de outra região)
    generate_refined_dem(
        "model/dem_refinement_unet.pth",
        "dados/ANADEM_Recorte_IPT_5m.tif",  # ANADEM da nova área
        None,  # GeoSampa da nova área (se disponível)
        "output/anadem_u_net.tif",
        tile_size=256,
        overlap=0.5
    )