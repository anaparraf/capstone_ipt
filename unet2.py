"""
U-Net Clássica com 2 Entradas (LIDAR + ANADEM) → 1 Saída (target refinado)
Todos os dados estão na MESMA resolução (5m).

Estrutura:
- Entrada: 2 canais (LIDAR reamostrado 0.5m→5m + ANADEM reamostrado 30m→5m)
- Saída: 1 canal (versão refinada em 5m)
- U-Net simétrica: 4 níveis encoder + bottleneck + 4 níveis decoder
"""

import os
import time
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import rasterio
from rasterio.windows import Window

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------------------
# Configuração
# ---------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")

# ---------------------------
# Métricas (PSNR mask-aware)
# ---------------------------
def calculate_psnr_masked(pred, target, mask, max_val=1.0):
    """PSNR considerando apenas pixels válidos (mask=1)"""
    B = pred.size(0)
    psnrs = []
    for b in range(B):
        m = mask[b:b+1].bool()
        if m.sum() == 0:
            continue
        p = pred[b:b+1][m]
        t = target[b:b+1][m]
        mse = torch.mean((p - t) ** 2)
        if mse == 0:
            psnrs.append(100.0)
        else:
            psnrs.append((20 * torch.log10(torch.tensor(max_val) / torch.sqrt(mse))).item())
    return float(np.mean(psnrs)) if psnrs else float('nan')

# ---------------------------
# Logger
# ---------------------------
class TrainingLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"training_{self.timestamp}.json"
        self.data = {
            'timestamp': self.timestamp,
            'config': {},
            'epochs': [],
            'metrics': {
                'train_loss': [], 'val_loss': [],
                'train_psnr': [], 'val_psnr': [],
                'learning_rate': []
            }
        }

    def set_config(self, config):
        self.data['config'] = config

    def log_epoch(self, epoch, train_loss, val_loss, train_psnr, val_psnr, lr):
        self.data['epochs'].append({
            'epoch': epoch, 'train_loss': float(train_loss),
            'val_loss': float(val_loss) if val_loss else None,
            'train_psnr': float(train_psnr) if train_psnr else None,
            'val_psnr': float(val_psnr) if val_psnr else None,
            'lr': float(lr)
        })
        self.data['metrics']['train_loss'].append(float(train_loss))
        self.data['metrics']['val_loss'].append(float(val_loss) if val_loss else float('nan'))
        self.data['metrics']['train_psnr'].append(float(train_psnr) if train_psnr else float('nan'))
        self.data['metrics']['val_psnr'].append(float(val_psnr) if val_psnr else float('nan'))
        self.data['metrics']['learning_rate'].append(float(lr))
        self._save()

    def _save(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.data, f, indent=2)

    def get_log_file(self):
        return str(self.log_file)

class TrainingVisualizer:
    @staticmethod
    def plot_curves(log_file, out_png=None):
        with open(log_file, 'r') as f:
            data = json.load(f)
        metrics = data['metrics']
        epochs = np.arange(1, len(metrics['train_loss']) + 1)

        fig = plt.figure(figsize=(14, 5))
        
        ax1 = plt.subplot(1, 2, 1)
        ax1.plot(epochs, metrics['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, metrics['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Época', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Loss de Treinamento')

        ax2 = plt.subplot(1, 2, 2)
        ax2.plot(epochs, metrics['train_psnr'], 'b-', label='Train PSNR', linewidth=2)
        ax2.plot(epochs, metrics['val_psnr'], 'r-', label='Val PSNR', linewidth=2)
        ax2.set_xlabel('Época', fontsize=12)
        ax2.set_ylabel('PSNR (dB)', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_title('PSNR de Validação')

        plt.tight_layout()
        if out_png is None:
            out_png = log_file.replace('.json', '_curves.png')
        plt.savefig(out_png, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Gráficos salvos: {out_png}")

# ---------------------------
# Dataset: 2 Entradas + 1 Target
# ---------------------------
class DualInputDataset(Dataset):
    """
    Carrega 3 rasters (todos na mesma resolução 5m):
    - anadem_files: ANADEM reamostrado 30m→5m (entrada 1)
    - lidar_files: LIDAR reamostrado 0.5m→5m (entrada 2)
    - target_files: Ground truth em 5m (saída esperada)
    """
    def __init__(self, anadem_files, lidar_files, target_files, 
                 patch_size=128, samples_per_file=1000):
        assert len(anadem_files) == len(lidar_files) == len(target_files)
        self.anadem = [rasterio.open(p) for p in anadem_files]
        self.lidar = [rasterio.open(p) for p in lidar_files]
        self.target = [rasterio.open(p) for p in target_files]
        self.patch_size = patch_size
        self.samples_per_file = samples_per_file
        
        print(f"Dataset inicializado com {len(anadem_files)} conjuntos de imagens")

    def __len__(self):
        return len(self.anadem) * self.samples_per_file

    def _normalize(self, img, nodata):
        """Normaliza para [0,1] usando percentis 2-98"""
        valid = np.isfinite(img)
        if nodata is not None:
            valid &= (img != nodata)
        
        if not np.any(valid):
            return np.zeros_like(img, dtype=np.float32), valid.astype(np.uint8)
        
        arr = img.copy().astype(np.float32)
        vals = arr[valid]
        p2, p98 = np.percentile(vals, [2, 98])
        
        if (p98 - p2) < 1e-6:
            p98 = p2 + 1e-6
            
        norm = np.zeros_like(arr, dtype=np.float32)
        norm[valid] = np.clip((arr[valid] - p2) / (p98 - p2), 0, 1)
        
        return norm, valid.astype(np.uint8)

    def __getitem__(self, idx):
        file_idx = idx % len(self.anadem)
        
        # Abre os 3 rasters
        anadem_src = self.anadem[file_idx]
        lidar_src = self.lidar[file_idx]
        target_src = self.target[file_idx]
        
        # Encontra dimensões mínimas entre os 3 rasters
        min_width = min(anadem_src.width, lidar_src.width, target_src.width)
        min_height = min(anadem_src.height, lidar_src.height, target_src.height)
        
        # Extrai patch aleatório (garante que cabe em todos os 3)
        max_x = max(0, min_width - self.patch_size)
        max_y = max(0, min_height - self.patch_size)
        
        if max_x == 0 or max_y == 0:
            # Se rasters são menores que patch_size, usa tamanho disponível
            rx, ry = 0, 0
            patch_w = min(min_width, self.patch_size)
            patch_h = min(min_height, self.patch_size)
        else:
            rx = np.random.randint(0, max_x + 1)
            ry = np.random.randint(0, max_y + 1)
            patch_w = self.patch_size
            patch_h = self.patch_size
        
        # Mesma janela para os 3 rasters
        window = Window(rx, ry, patch_w, patch_h)
        
        # Lê dados
        try:
            anadem = anadem_src.read(1, window=window).astype(np.float32)
            lidar = lidar_src.read(1, window=window).astype(np.float32)
            target = target_src.read(1, window=window).astype(np.float32)
        except Exception as e:
            print(f"[WARN] Erro ao ler window {window}: {e}")
            # Fallback: lê região top-left
            anadem = anadem_src.read(1).astype(np.float32)[:self.patch_size, :self.patch_size]
            lidar = lidar_src.read(1).astype(np.float32)[:self.patch_size, :self.patch_size]
            target = target_src.read(1).astype(np.float32)[:self.patch_size, :self.patch_size]
        
        # CRÍTICO: Garante que todos têm EXATAMENTE o mesmo shape
        # Se tiverem shapes diferentes, resample para o menor comum
        shapes = [anadem.shape, lidar.shape, target.shape]
        if len(set(shapes)) > 1:
            print(f"[WARN] Shapes diferentes detectados: {shapes}")
            # Encontra menor shape comum
            min_h = min(s[0] for s in shapes)
            min_w = min(s[1] for s in shapes)
            
            # Crop todos para mesmo tamanho
            anadem = anadem[:min_h, :min_w]
            lidar = lidar[:min_h, :min_w]
            target = target[:min_h, :min_w]
        
        # Padding se necessário (garante patch_size x patch_size)
        if anadem.shape[0] < self.patch_size or anadem.shape[1] < self.patch_size:
            anadem = np.pad(anadem, 
                          ((0, max(0, self.patch_size - anadem.shape[0])),
                           (0, max(0, self.patch_size - anadem.shape[1]))),
                          mode='constant', constant_values=0)
            lidar = np.pad(lidar,
                          ((0, max(0, self.patch_size - lidar.shape[0])),
                           (0, max(0, self.patch_size - lidar.shape[1]))),
                          mode='constant', constant_values=0)
            target = np.pad(target,
                           ((0, max(0, self.patch_size - target.shape[0])),
                            (0, max(0, self.patch_size - target.shape[1]))),
                           mode='constant', constant_values=0)
        
        # Normaliza
        anadem_n, mask_a = self._normalize(anadem, anadem_src.nodata)
        lidar_n, mask_l = self._normalize(lidar, lidar_src.nodata)
        target_n, mask_t = self._normalize(target, target_src.nodata)
        
        # Verifica shapes das masks ANTES de combinar
        assert mask_a.shape == mask_l.shape == mask_t.shape, \
            f"Mask shapes não batem: {mask_a.shape}, {mask_l.shape}, {mask_t.shape}"
        
        # Mask final: interseção dos 3
        mask = (mask_a & mask_l & mask_t).astype(np.float32)
        
        # Converte para tensores: (1,H,W) cada
        anadem_t = torch.from_numpy(anadem_n).unsqueeze(0)
        lidar_t = torch.from_numpy(lidar_n).unsqueeze(0)
        target_t = torch.from_numpy(target_n).unsqueeze(0)
        mask_t = torch.from_numpy(mask).unsqueeze(0)
        
        # Concatena as 2 entradas: (2,H,W)
        inputs = torch.cat([anadem_t, lidar_t], dim=0)
        
        return inputs, target_t, mask_t

    def __del__(self):
        for src in self.anadem + self.lidar + self.target:
            try:
                src.close()
            except:
                pass

# ---------------------------
# Modelo: U-Net Clássica Simétrica
# ---------------------------
class DoubleConv(nn.Module):
    """Bloco clássico da U-Net: Conv-BN-ReLU-Conv-BN-ReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """
    U-Net Clássica Simétrica
    - in_channels: número de canais de entrada (2 para ANADEM+LIDAR)
    - out_channels: número de canais de saída (1 para target)
    - features: lista de canais em cada nível [64, 128, 256, 512]
    """
    def __init__(self, in_channels=2, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # ENCODER (Downsampling)
        in_ch = in_channels
        for feature in features:
            self.downs.append(DoubleConv(in_ch, feature))
            in_ch = feature
        
        # BOTTLENECK
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        
        # DECODER (Upsampling)
        for feature in reversed(features):
            # Transposed conv para upsampling
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            # DoubleConv após concatenação com skip connection
            # Entrada: feature (up) + feature (skip) = feature * 2
            self.ups.append(DoubleConv(feature * 2, feature))
        
        # Camada final: 1x1 conv
        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
        # Inicialização He
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        skip_connections = []
        
        # ENCODER
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # BOTTLENECK
        x = self.bottleneck(x)
        
        # DECODER
        skip_connections = skip_connections[::-1]  # inverte ordem
        
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # upsampling
            skip = skip_connections[idx // 2]
            
            # Garante mesmo tamanho (por causa de padding)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            
            # Concatena com skip connection
            x = torch.cat([skip, x], dim=1)
            x = self.ups[idx + 1](x)  # double conv
        
        # Camada final
        return self.final(x)

# ---------------------------
# Loss Function (MSE + Gradient)
# ---------------------------
class CombinedLoss(nn.Module):
    """MSE + Gradient Loss, ambos mask-aware"""
    def __init__(self, alpha=0.7, beta=0.3):
        super().__init__()
        self.alpha = alpha  # peso do MSE
        self.beta = beta    # peso do gradiente
    
    def forward(self, pred, target, mask):
        # MSE mask-aware
        mask_count = torch.clamp(mask.sum(), min=1.0)
        mse = (((pred - target) ** 2) * mask).sum() / mask_count
        
        # Gradient loss mask-aware
        grad_loss = self._gradient_loss(pred, target, mask)
        
        return self.alpha * mse + self.beta * grad_loss
    
    def _gradient_loss(self, pred, target, mask):
        """Loss baseado em diferenças de gradiente"""
        # Gradientes em x e y
        pred_gx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_gy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        
        target_gx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_gy = target[:, :, 1:, :] - target[:, :, :-1, :]
        
        # Masks para gradientes
        mask_gx = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        mask_gy = mask[:, :, 1:, :] * mask[:, :, :-1, :]
        
        # Loss
        loss_gx = (torch.abs(pred_gx - target_gx) * mask_gx).sum()
        loss_gy = (torch.abs(pred_gy - target_gy) * mask_gy).sum()
        
        count_gx = torch.clamp(mask_gx.sum(), min=1.0)
        count_gy = torch.clamp(mask_gy.sum(), min=1.0)
        
        return (loss_gx / count_gx + loss_gy / count_gy) / 2.0

# ---------------------------
# Validação de Dados
# ---------------------------
def validate_raster_compatibility(anadem_files, lidar_files, target_files):
    """Valida que os rasters são compatíveis antes do treino"""
    print("\n" + "="*60)
    print("VALIDANDO COMPATIBILIDADE DOS RASTERS")
    print("="*60)
    
    for i, (a_path, l_path, t_path) in enumerate(zip(anadem_files, lidar_files, target_files)):
        print(f"\nConjunto {i+1}:")
        print(f"  ANADEM: {a_path}")
        print(f"  LIDAR:  {l_path}")
        print(f"  TARGET: {t_path}")
        
        with rasterio.open(a_path) as a_src, \
             rasterio.open(l_path) as l_src, \
             rasterio.open(t_path) as t_src:
            
            # Dimensões
            print(f"\n  Dimensões:")
            print(f"    ANADEM: {a_src.width} x {a_src.height}")
            print(f"    LIDAR:  {l_src.width} x {l_src.height}")
            print(f"    TARGET: {t_src.width} x {t_src.height}")
            
            # Resolução
            a_res = (abs(a_src.transform.a), abs(a_src.transform.e))
            l_res = (abs(l_src.transform.a), abs(l_src.transform.e))
            t_res = (abs(t_src.transform.a), abs(t_src.transform.e))
            
            print(f"\n  Resolução (x, y):")
            print(f"    ANADEM: {a_res[0]:.2f}m x {a_res[1]:.2f}m")
            print(f"    LIDAR:  {l_res[0]:.2f}m x {l_res[1]:.2f}m")
            print(f"    TARGET: {t_res[0]:.2f}m x {t_res[1]:.2f}m")
            
            # CRS
            print(f"\n  Sistema de Coordenadas:")
            print(f"    ANADEM: {a_src.crs}")
            print(f"    LIDAR:  {l_src.crs}")
            print(f"    TARGET: {t_src.crs}")
            
            # Warnings
            if a_src.crs != l_src.crs or a_src.crs != t_src.crs:
                print(f"  ⚠️  AVISO: CRS diferentes! Considere reprojetar.")
            
            if not (4.5 <= a_res[0] <= 5.5 and 4.5 <= l_res[0] <= 5.5):
                print(f"  ⚠️  AVISO: Resolução não está próxima de 5m!")
            
            # Sobreposição espacial
            a_bounds = a_src.bounds
            l_bounds = l_src.bounds
            t_bounds = t_src.bounds
            
            def bounds_overlap(b1, b2):
                return not (b1.right < b2.left or b1.left > b2.right or
                          b1.top < b2.bottom or b1.bottom > b2.top)
            
            if not (bounds_overlap(a_bounds, l_bounds) and bounds_overlap(a_bounds, t_bounds)):
                print(f"  ⚠️  AVISO: Rasters podem não ter sobreposição espacial!")
    
    print("\n" + "="*60)
    print("Validação concluída. Verifique os avisos acima.")
    print("="*60 + "\n")

# ---------------------------
# Treinamento
# ---------------------------
def train_model(
    train_anadem, train_lidar, train_target,
    val_anadem, val_lidar, val_target,
    epochs=50, batch_size=4, patch_size=128,
    lr=1e-4, save_path="model/unet_dual_input.pth"
):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    
    logger = TrainingLogger()
    
    # Datasets
    train_ds = DualInputDataset(train_anadem, train_lidar, train_target, 
                                 patch_size=patch_size, samples_per_file=1000)
    val_ds = DualInputDataset(val_anadem, val_lidar, val_target,
                               patch_size=patch_size, samples_per_file=200)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                             num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                           num_workers=0, pin_memory=True)
    
    # Modelo
    model = UNet(in_channels=2, out_channels=1, features=[64, 128, 256, 512]).to(device)
    
    # Otimizador e scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    criterion = CombinedLoss(alpha=0.7, beta=0.3).to(device)
    
    config = {
        'epochs': epochs, 'batch_size': batch_size, 'patch_size': patch_size,
        'lr': lr, 'in_channels': 2, 'out_channels': 1
    }
    logger.set_config(config)
    
    best_val_psnr = -np.inf
    
    print(f"Iniciando treinamento: {epochs} épocas")
    print(f"Modelo: U-Net com 2 entradas (ANADEM+LIDAR) → 1 saída")
    print("="*60)
    
    for epoch in range(1, epochs + 1):
        # TREINO
        model.train()
        train_loss = 0.0
        train_psnr = 0.0
        n_batches = 0
        
        for inputs, target, mask in train_loader:
            inputs = inputs.to(device)
            target = target.to(device)
            mask = mask.to(device)
            
            if mask.sum() < 1.0:
                continue
            
            optimizer.zero_grad()
            pred = model(inputs)
            loss = criterion(pred, target, mask)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[WARN] Epoch {epoch}: Loss inválido, pulando batch")
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_psnr += calculate_psnr_masked(pred, target, mask)
            n_batches += 1
        
        avg_train_loss = train_loss / n_batches if n_batches > 0 else float('nan')
        avg_train_psnr = train_psnr / n_batches if n_batches > 0 else float('nan')
        
        # VALIDAÇÃO
        model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        n_val = 0
        
        with torch.no_grad():
            for inputs, target, mask in val_loader:
                inputs = inputs.to(device)
                target = target.to(device)
                mask = mask.to(device)
                
                if mask.sum() < 1.0:
                    continue
                
                pred = model(inputs)
                loss = criterion(pred, target, mask)
                
                val_loss += loss.item()
                val_psnr += calculate_psnr_masked(pred, target, mask)
                n_val += 1
        
        avg_val_loss = val_loss / n_val if n_val > 0 else float('nan')
        avg_val_psnr = val_psnr / n_val if n_val > 0 else float('nan')
        
        # Scheduler
        scheduler.step(avg_val_loss if np.isfinite(avg_val_loss) else float('inf'))
        
        # Log
        lr_now = optimizer.param_groups[0]['lr']
        logger.log_epoch(epoch, avg_train_loss, avg_val_loss, 
                        avg_train_psnr, avg_val_psnr, lr_now)
        
        print(f"Época {epoch}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Train PSNR: {avg_train_psnr:.2f} dB | Val PSNR: {avg_val_psnr:.2f} dB | "
              f"LR: {lr_now:.2e}")
        
        # Salva melhor modelo
        if np.isfinite(avg_val_psnr) and avg_val_psnr > best_val_psnr:
            best_val_psnr = avg_val_psnr
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_psnr': best_val_psnr,
                'config': config
            }, save_path)
            print(f"✓ Melhor modelo salvo! PSNR: {best_val_psnr:.2f} dB")
    
    print("="*60)
    print("Treinamento finalizado!")
    return model, logger.get_log_file()

# ---------------------------
# Inferência
# ---------------------------
def generate_output(model_path, anadem_path, lidar_path, output_path, 
                   tile_size=256, overlap=0.5):
    """
    Gera raster de saída usando modelo treinado
    """
    # Carrega modelo
    ckpt = torch.load(model_path, map_location=device)
    model = UNet(in_channels=2, out_channels=1).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    print(f"Modelo carregado: {model_path}")
    print(f"Val PSNR do checkpoint: {ckpt.get('val_psnr', 'N/A'):.2f} dB")
    
    # Abre rasters de entrada
    with rasterio.open(anadem_path) as anadem_src, \
         rasterio.open(lidar_path) as lidar_src:
        
        # Lê e normaliza
        anadem = anadem_src.read(1).astype(np.float32)
        lidar = lidar_src.read(1).astype(np.float32)
        profile = anadem_src.profile.copy()
        
        # Normalização simples (usar mesma lógica do treino)
        def normalize(img, nodata):
            valid = np.isfinite(img)
            if nodata is not None:
                valid &= (img != nodata)
            if not np.any(valid):
                # retorna zeros + estatísticas neutras
                return np.zeros_like(img, dtype=np.float32), 0.0, 1.0
            arr = img.copy().astype(np.float32)
            p2, p98 = np.percentile(arr[valid], [2, 98])
            if (p98 - p2) < 1e-6:
                p98 = p2 + 1e-6
            norm = np.zeros_like(arr, dtype=np.float32)
            norm[valid] = np.clip((arr[valid] - p2) / (p98 - p2), 0, 1)
            return norm, p2, p98
        
        anadem_n, a_p2, a_p98 = normalize(anadem, anadem_src.nodata)
        lidar_n, l_p2, l_p98 = normalize(lidar, lidar_src.nodata)
        
        # Prepara output
        h, w = anadem_n.shape
        output = np.zeros((h, w), dtype=np.float32)
        weight = np.zeros((h, w), dtype=np.float32)
        
        # Hanning window para blending suave
        hann = np.outer(np.hanning(tile_size), np.hanning(tile_size))
        
        # Processa em tiles
        step = int(tile_size * (1 - overlap))
        total_tiles = ((h - 1) // step + 1) * ((w - 1) // step + 1)
        processed = 0
        
        print(f"Processando {total_tiles} tiles...")
        
        for i in range(0, h, step):
            for j in range(0, w, step):
                i_end = min(i + tile_size, h)
                j_end = min(j + tile_size, w)
                
                # Extrai tiles (podem ter shapes diferentes entre os dois rasters)
                tile_a = anadem_n[i:i_end, j:j_end]
                tile_l = lidar_n[i:i_end, j:j_end]
                
                # Garante menor shape comum entre os dois canais (evita broadcast error)
                th_a, tw_a = tile_a.shape
                th_l, tw_l = tile_l.shape
                th = min(th_a, th_l)
                tw = min(tw_a, tw_l)
                
                # Crop para o menor shape comum
                tile_a = tile_a[:th, :tw]
                tile_l = tile_l[:th, :tw]
                
                # Padding se necessário para atingir tile_size
                if th < tile_size or tw < tile_size:
                    pad_a = np.zeros((tile_size, tile_size), dtype=np.float32)
                    pad_l = np.zeros((tile_size, tile_size), dtype=np.float32)
                    pad_a[:th, :tw] = tile_a
                    pad_l[:th, :tw] = tile_l
                    tile_a, tile_l = pad_a, pad_l
                else:
                    # já tem tamanho completo
                    pass
                
                # Concatena canais: (2,H,W) -> (1,2,H,W)
                tile = np.stack([tile_a, tile_l], axis=0)
                tile_tensor = torch.from_numpy(tile).unsqueeze(0).to(device)
                
                # Inferência
                with torch.no_grad():
                    out_tile = model(tile_tensor).cpu().numpy().squeeze()
                
                # Usa apenas a região válida (antes do padding)
                out_valid = out_tile[:th, :tw]
                w_valid = hann[:th, :tw]
                
                # Escreve na posição correta (usa th/tw reais)
                write_i_end = i + th
                write_j_end = j + tw
                output[i:write_i_end, j:write_j_end] += out_valid * w_valid
                weight[i:write_i_end, j:write_j_end] += w_valid
                
                processed += 1
                if processed % 100 == 0:
                    print(f"  {processed}/{total_tiles} tiles processados")
        
        # Normalização final
        valid_w = weight > 1e-6
        output[valid_w] /= weight[valid_w]
        
        # Desnormaliza (usa estatísticas do LIDAR como referência)
        output = output * (l_p98 - l_p2) + l_p2
        
        # Salva
        profile.update(dtype='float32', count=1)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(output.astype(np.float32), 1)
        
        print(f"✓ Raster salvo: {output_path}")

# ---------------------------
# Execução
# ---------------------------
if __name__ == "__main__":
    # AJUSTE OS CAMINHOS DOS SEUS DADOS AQUI
    # =============================================
    # IMPORTANTE: Seus dados atuais
    # =============================================
    
    # Treino
    train_anadem_files = ["dados/anadem_5m.tif"]       # ANADEM reamostrado para 5m
    train_lidar_files = ["dados/geosampa_30m_reamostrado_5m.tif"]      # Usando GEOSAMPA como "LIDAR"
    train_target_files = ["dados/MDTGeosampa_AricanduvaBufferUTM.tif"]     # Ground truth em 5m
    
    # Validação (região diferente)
    val_anadem_files = ["dados/ANADEM_Recorte_IPT_5m.tif"]
    val_lidar_files = ["dados/GEOSAMPA_Recorte_IPT_reamostrado_5m.tif"]   # Target também serve como input
    val_target_files = ["dados/GEOSAMPA_Recorte_IPT_reprojetado.tif"]
    
    # Cria diretórios
    os.makedirs("model", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # VALIDA COMPATIBILIDADE DOS DADOS
    print("\n🔍 Verificando dados de TREINO:")
    validate_raster_compatibility(train_anadem_files, train_lidar_files, train_target_files)
    
    print("\n🔍 Verificando dados de VALIDAÇÃO:")
    validate_raster_compatibility(val_anadem_files, val_lidar_files, val_target_files)
    
    print("="*60)
    print("TREINAMENTO U-NET DUAL-INPUT")
    print("="*60)
    print("Entradas: 2 canais (ANADEM 5m + LIDAR 5m)")
    print("Saída: 1 canal (versão refinada em 5m)")
    print("="*60)
    
    # Treina modelo
    model, log_file = train_model(
        train_anadem=train_anadem_files,
        train_lidar=train_lidar_files,
        train_target=train_target_files,
        val_anadem=val_anadem_files,
        val_lidar=val_lidar_files,
        val_target=val_target_files,
        epochs=90,           # Ajuste conforme necessário
        batch_size=4,        # Ajuste conforme sua VRAM (4-8 é seguro)
        patch_size=64,      # 128 ou 256
        lr=1e-4,             # Learning rate inicial
        save_path="model/unet_dual_input_best.pth"
    )
    
    print("\n" + "="*60)
    print("GERANDO GRÁFICOS DE TREINAMENTO")
    print("="*60)
    TrainingVisualizer.plot_curves(log_file)
    
    print("\n" + "="*60)
    print("INFERÊNCIA - GERANDO RASTER MELHORADO")
    print("="*60)
    
    # Verifica se modelo foi salvo
    model_path = "model/unet_dual_input_best.pth"
    if os.path.exists(model_path):
        # Gera output para região de validação
        generate_output(
            model_path=model_path,
            anadem_path="dados/ANADEM_Recorte_IPT_5m.tif",
            lidar_path="dados/GEOSAMPA_Recorte_IPT_reamostrado_5m.tif",
            output_path="output/resultado_unet_dual_input.tif",
            tile_size=256,
            overlap=0.5  # 50% overlap para blending suave
        )
        print("\n✓ Pipeline completo finalizado!")
        print(f"  - Modelo: {model_path}")
        print(f"  - Logs: {log_file}")
        print(f"  - Output: output/resultado_unet_dual_input_90.tif")
    else:
        print(f"[ERRO] Modelo não encontrado: {model_path}")
        print("Execute o treinamento primeiro.")
    
    print("="*60)
    print("FIM DA EXECUÇÃO")
    print("="*60)


# =============================================
# INSTRUÇÕES DE USO
# =============================================
"""
1. PREPARAÇÃO DOS DADOS:
   Antes de rodar este script, você precisa reamostrar seus dados para 5m:
   
   a) ANADEM (30m → 5m):
      gdal_translate -tr 5 5 -r bilinear anadem_30m.tif anadem_5m.tif
   
   b) LIDAR (0.5m → 5m):
      gdal_translate -tr 5 5 -r average lidar_0.5m.tif lidar_5m.tif
   
   c) Target ground truth já deve estar em 5m (geosampa_5m.tif)

2. ESTRUTURA DE PASTAS:
   projeto/
   ├── dados/
   │   ├── anadem_5m.tif              # ANADEM reamostrado
   │   ├── lidar_5m.tif               # LIDAR reamostrado
   │   ├── geosampa_5m.tif            # Target (ground truth)
   │   ├── ANADEM_Recorte_IPT_5m.tif  # Validação
   │   ├── LIDAR_Recorte_IPT_5m.tif   # Validação
   │   └── GEOSAMPA_Recorte_IPT.tif   # Validação target
   ├── model/                         # Checkpoints salvos aqui
   ├── output/                        # Rasters gerados aqui
   └── logs/                          # JSONs de treinamento

3. AJUSTAR HIPERPARÂMETROS:
   - batch_size: depende da VRAM (4-8 é seguro, 16+ se tiver muita VRAM)
   - patch_size: 128 ou 256 (maior = mais contexto, mas mais VRAM)
   - epochs: 30-100 (ajuste conforme convergência)
   - lr: 1e-4 é bom ponto de partida

4. MONITORAMENTO:
   - Logs salvos em logs/training_TIMESTAMP.json
   - Gráficos salvos automaticamente como PNG
   - Modelo salvo quando val_psnr melhorar

5. INFERÊNCIA:
   Após treinar, o script automaticamente gera o raster melhorado.
   Para gerar para outras regiões, use a função generate_output()

6. TROUBLESHOOTING:
   - Se der OOM (out of memory): reduza batch_size ou patch_size
   - Se loss=NaN: verifique normalização dos dados (nodata, infinitos)
   - Se PSNR não melhora: aumente epochs, ajuste lr, ou verifique dados
"""