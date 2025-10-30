import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from datetime import datetime
import rasterio
from rasterio.windows import Window

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")


# ============================================================================
# ARQUITETURA U-NET CORRIGIDA
# ============================================================================

class DoubleConv(nn.Module):
    """Bloco de dupla convolução: (Conv2D -> BatchNorm -> ReLU) x 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Bloco de downsampling: MaxPool2D -> DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Bloco de upsampling: ConvTranspose2D -> Concatenar skip -> DoubleConv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Garante que as dimensões batem para concatenação
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Convolução final 1x1 para gerar a saída"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNetRefinement(nn.Module):
    """
    U-Net Clássica para Refinamento de DEM
    Entrada e saída têm a mesma resolução espacial
    """
    def __init__(self, n_channels=1, n_classes=1, base_filters=64, bilinear=True):
        super(UNetRefinement, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder
        self.inc = DoubleConv(n_channels, base_filters)
        self.down1 = Down(base_filters, base_filters * 2)
        self.down2 = Down(base_filters * 2, base_filters * 4)
        self.down3 = Down(base_filters * 4, base_filters * 8)
        
        # Bottleneck
        factor = 2 if bilinear else 1
        self.down4 = Down(base_filters * 8, base_filters * 16 // factor)
        
        # Decoder
        self.up1 = Up(base_filters * 16, base_filters * 8 // factor, bilinear)
        self.up2 = Up(base_filters * 8, base_filters * 4 // factor, bilinear)
        self.up3 = Up(base_filters * 4, base_filters * 2 // factor, bilinear)
        self.up4 = Up(base_filters * 2, base_filters, bilinear)
        
        # Saída
        self.outc = OutConv(base_filters, n_classes)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder com skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Saída
        logits = self.outc(x)
        return logits


# ============================================================================
# FUNÇÕES DE PERDA
# ============================================================================

class PerceptualLossDEM(nn.Module):
    """Loss combinada para refinamento de DEM"""
    def __init__(self, alpha_mse=0.5, alpha_grad=0.3, alpha_ssim=0.2):
        super().__init__()
        self.alpha_mse = alpha_mse
        self.alpha_grad = alpha_grad
        self.alpha_ssim = alpha_ssim
        
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
    
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        grad_loss = self._gradient_loss(pred, target)
        ssim_loss = 1 - self._ssim(pred, target)
        
        total_loss = (self.alpha_mse * mse_loss + 
                     self.alpha_grad * grad_loss + 
                     self.alpha_ssim * ssim_loss)
        
        return total_loss, mse_loss, grad_loss, ssim_loss
    
    def _gradient_loss(self, pred, target):
        def get_gradients(img):
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                   dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                   dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
            
            grad_x = F.conv2d(img, sobel_x, padding=1)
            grad_y = F.conv2d(img, sobel_y, padding=1)
            
            return grad_x, grad_y
        
        pred_gx, pred_gy = get_gradients(pred)
        target_gx, target_gy = get_gradients(target)
        
        loss = self.l1(pred_gx, target_gx) + self.l1(pred_gy, target_gy)
        return loss
    
    def _ssim(self, pred, target, window_size=11):
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


def calculate_psnr(pred, target, max_val=1.0):
    """Calcula PSNR entre predição e target"""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return 100.0
    psnr = 20 * torch.log10(torch.tensor(max_val) / torch.sqrt(mse))
    return psnr.item()


# ============================================================================
# LOGGER
# ============================================================================

class TrainingLogger:
    """Logger para salvar métricas durante treinamento"""
    
    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"training_{self.timestamp}.json"
        
        self.data = {
            'timestamp': self.timestamp,
            'config': {},
            'epochs': [],
            'batches': [],
            'metrics': {
                'epoch_loss': [],
                'mse_loss': [],
                'grad_loss': [],
                'ssim_loss': [],
                'learning_rate': [],
                'psnr': [],
            }
        }
        
        print(f"Logger inicializado: {self.log_file}")
    
    def set_config(self, config_dict):
        self.data['config'] = config_dict
    
    def log_batch(self, epoch, batch_idx, loss, mse, grad, ssim, lr):
        self.data['batches'].append({
            'epoch': epoch,
            'batch': batch_idx,
            'loss': float(loss),
            'mse': float(mse),
            'grad': float(grad),
            'ssim': float(ssim),
            'lr': float(lr),
        })
    
    def log_epoch(self, epoch, epoch_loss, mse, grad, ssim, lr, psnr=None):
        self.data['epochs'].append({
            'epoch': epoch,
            'loss': float(epoch_loss),
            'mse': float(mse),
            'grad': float(grad),
            'ssim': float(ssim),
            'lr': float(lr),
            'psnr': float(psnr) if psnr is not None else None,
        })
        
        self.data['metrics']['epoch_loss'].append(float(epoch_loss))
        self.data['metrics']['mse_loss'].append(float(mse))
        self.data['metrics']['grad_loss'].append(float(grad))
        self.data['metrics']['ssim_loss'].append(float(ssim))
        self.data['metrics']['learning_rate'].append(float(lr))
        
        if psnr is not None:
            self.data['metrics']['psnr'].append(float(psnr))
        
        self.save()
    
    def save(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def get_log_file(self):
        return str(self.log_file)


# ============================================================================
# VISUALIZADOR
# ============================================================================

class TrainingVisualizer:
    """Plota gráficos de performance do treinamento"""
    
    @staticmethod
    def load_log(log_file):
        with open(log_file, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def plot_training_results(log_file, output_file=None):
        data = TrainingVisualizer.load_log(log_file)
        metrics = data['metrics']
        
        if not metrics['epoch_loss']:
            print("Nenhuma métrica para plotar!")
            return
        
        epochs = range(1, len(metrics['epoch_loss']) + 1)
        
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)
        
        # Loss Total
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(epochs, metrics['epoch_loss'], 'b-', linewidth=2, label='Total Loss')
        ax1.fill_between(epochs, metrics['epoch_loss'], alpha=0.3)
        ax1.set_xlabel('Época', fontsize=11)
        ax1.set_ylabel('Loss', fontsize=11)
        ax1.set_title('Loss Total por Época', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # MSE Loss
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(epochs, metrics['mse_loss'], 'g-', linewidth=2, label='MSE Loss')
        ax2.fill_between(epochs, metrics['mse_loss'], alpha=0.3, color='green')
        ax2.set_xlabel('Época', fontsize=11)
        ax2.set_ylabel('MSE Loss', fontsize=11)
        ax2.set_title('Mean Squared Error', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Gradient Loss
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(epochs, metrics['grad_loss'], 'r-', linewidth=2, label='Gradient Loss')
        ax3.fill_between(epochs, metrics['grad_loss'], alpha=0.3, color='red')
        ax3.set_xlabel('Época', fontsize=11)
        ax3.set_ylabel('Gradient Loss', fontsize=11)
        ax3.set_title('Preservação de Bordas', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # SSIM Loss
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(epochs, metrics['ssim_loss'], 'purple', linewidth=2, label='SSIM Loss')
        ax4.fill_between(epochs, metrics['ssim_loss'], alpha=0.3, color='purple')
        ax4.set_xlabel('Época', fontsize=11)
        ax4.set_ylabel('SSIM Loss', fontsize=11)
        ax4.set_title('Similaridade Estrutural', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Learning Rate
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(epochs, metrics['learning_rate'], 'brown', linewidth=2, label='Learning Rate')
        ax5.fill_between(epochs, metrics['learning_rate'], alpha=0.3, color='brown')
        ax5.set_xlabel('Época', fontsize=11)
        ax5.set_ylabel('Learning Rate', fontsize=11)
        ax5.set_title('Taxa de Aprendizado', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.set_yscale('log')
        ax5.legend()
        
        # PSNR
        if metrics.get('psnr') and len(metrics['psnr']) > 0:
            ax6 = fig.add_subplot(gs[2, 1])
            ax6.plot(epochs[:len(metrics['psnr'])], metrics['psnr'], 'darkgreen', 
                    linewidth=2.5, marker='o', markersize=5, label='PSNR')
            ax6.fill_between(epochs[:len(metrics['psnr'])], metrics['psnr'], alpha=0.3, color='darkgreen')
            ax6.set_xlabel('Época', fontsize=11)
            ax6.set_ylabel('PSNR (dB)', fontsize=11)
            ax6.set_title('Peak Signal-to-Noise Ratio', fontsize=12, fontweight='bold')
            ax6.grid(True, alpha=0.3)
            ax6.legend()
        
        plt.suptitle('Monitoramento de Treinamento - U-Net Refinamento DEM', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        if output_file is None:
            output_file = log_file.replace('.json', '_plots.png')
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Gráficos salvos em: {output_file}")
        plt.show()
        
        return output_file
    
    @staticmethod
    def print_summary(log_file):
        data = TrainingVisualizer.load_log(log_file)
        metrics = data['metrics']
        
        if not metrics['epoch_loss']:
            print("Nenhuma métrica para resumir!")
            return
        
        print("\n" + "="*60)
        print("RESUMO DE TREINAMENTO")
        print("="*60)
        print(f"Timestamp: {data['timestamp']}")
        print(f"Total de Épocas: {len(metrics['epoch_loss'])}")
        print()
        
        print("LOSS TOTAL")
        print(f"  Inicial: {metrics['epoch_loss'][0]:.6f}")
        print(f"  Final:   {metrics['epoch_loss'][-1]:.6f}")
        print(f"  Mínimo:  {min(metrics['epoch_loss']):.6f} (época {np.argmin(metrics['epoch_loss']) + 1})")
        print(f"  Redução: {(metrics['epoch_loss'][0] - metrics['epoch_loss'][-1]) / metrics['epoch_loss'][0] * 100:.2f}%")
        print()
        
        print("COMPONENTES DA LOSS")
        print(f"  MSE Loss:      {metrics['mse_loss'][-1]:.6f} (init: {metrics['mse_loss'][0]:.6f})")
        print(f"  Gradient Loss: {metrics['grad_loss'][-1]:.6f} (init: {metrics['grad_loss'][0]:.6f})")
        print(f"  SSIM Loss:     {metrics['ssim_loss'][-1]:.6f} (init: {metrics['ssim_loss'][0]:.6f})")
        print()
        
        if metrics.get('psnr') and len(metrics['psnr']) > 0:
            print("PSNR")
            print(f"  Inicial: {metrics['psnr'][0]:.2f} dB")
            print(f"  Final:   {metrics['psnr'][-1]:.2f} dB")
            print(f"  Máximo:  {max(metrics['psnr']):.2f} dB (época {np.argmax(metrics['psnr']) + 1})")
            print()
        
        print("LEARNING RATE")
        print(f"  Inicial: {metrics['learning_rate'][0]:.2e}")
        print(f"  Final:   {metrics['learning_rate'][-1]:.2e}")
        print("="*60 + "\n")


# ============================================================================
# DATASET
# ============================================================================

class DEMRefinementDataset(Dataset):
    """Dataset para refinamento de DEM - MESMA RESOLUÇÃO"""
    def __init__(self, low_quality_files, high_quality_files, patch_size=128, transform=None):
        self.low_quality_datasets = [rasterio.open(f) for f in low_quality_files]
        self.high_quality_datasets = [rasterio.open(f) for f in high_quality_files]
        self.transform = transform
        self.patch_size = patch_size
        
        assert len(self.low_quality_datasets) == len(self.high_quality_datasets)
        
        print(f"Dataset carregado com {len(self.low_quality_datasets)} arquivos")
        print(f"Patch size: {patch_size}x{patch_size}")
    
    def __len__(self):
        return 10000  # Número de patches por época
    
    def __getitem__(self, idx):
        # Escolhe arquivo aleatório
        file_idx = np.random.randint(0, len(self.low_quality_datasets))
        low_src = self.low_quality_datasets[file_idx]
        high_src = self.high_quality_datasets[file_idx]
        
        # Extrai patch aleatório
        max_x = low_src.width - self.patch_size
        max_y = low_src.height - self.patch_size
        
        if max_x <= 0 or max_y <= 0:
            window = Window(0, 0, min(low_src.width, self.patch_size), 
                          min(low_src.height, self.patch_size))
        else:
            random_x = np.random.randint(0, max_x)
            random_y = np.random.randint(0, max_y)
            window = Window(random_x, random_y, self.patch_size, self.patch_size)
        
        # Lê dados
        try:
            img_low = low_src.read(1, window=window).astype(np.float32)
            img_high = high_src.read(1, window=window).astype(np.float32)
        except Exception as e:
            print(f"Erro ao ler patch: {e}")
            img_low = low_src.read(1).astype(np.float32)[:self.patch_size, :self.patch_size]
            img_high = high_src.read(1).astype(np.float32)[:self.patch_size, :self.patch_size]
        
        # Normaliza
        img_low_norm = self._normalize_image(img_low, low_src.nodata)
        img_high_norm = self._normalize_image(img_high, high_src.nodata)
        
        # Garante mesmas dimensões
        if img_low_norm.shape != img_high_norm.shape:
            target_h, target_w = img_low_norm.shape
            img_high_norm_tensor = torch.from_numpy(img_high_norm).unsqueeze(0).unsqueeze(0)
            img_high_norm_tensor = F.interpolate(img_high_norm_tensor, size=(target_h, target_w),
                                                mode='bilinear', align_corners=False)
            img_high_norm = img_high_norm_tensor.squeeze().numpy()
        
        # Converte para tensor
        low_tensor = torch.from_numpy(img_low_norm).unsqueeze(0)
        high_tensor = torch.from_numpy(img_high_norm).unsqueeze(0)
        
        if self.transform:
            low_tensor = self.transform(low_tensor)
            high_tensor = self.transform(high_tensor)
        
        return low_tensor, high_tensor
    
    def _normalize_image(self, img, nodata_value):
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
            
            # Data augmentation: adiciona ruído leve
            if np.random.random() > 0.5:
                noise = np.random.normal(0, 0.02, img_norm.shape).astype(np.float32)
                img_norm = np.clip(img_norm + noise, 0, 1)
            
            return img_norm
        else:
            return np.zeros_like(img, dtype=np.float32)
    
    def __del__(self):
        for dataset in self.low_quality_datasets + self.high_quality_datasets:
            if hasattr(dataset, 'close'):
                dataset.close()


# ============================================================================
# TREINAMENTO
# ============================================================================

def train_unet_refinement(low_quality_files, high_quality_files,
                         epochs=50, batch_size=4, learning_rate=1e-4, 
                         patch_size=128, base_filters=64,
                         save_path="model/unet_refinement.pth"):
    
    print(f"\nIniciando treinamento de refinamento DEM")
    print(f"Patch size: {patch_size}x{patch_size}")
    print(f"Base filters: {base_filters}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    
    start_time = time.time()
    
    # Logger
    logger = TrainingLogger(log_dir="logs")
    
    # Dataset e DataLoader
    dataset = DEMRefinementDataset(
        low_quality_files, high_quality_files,
        patch_size=patch_size
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                           num_workers=0, pin_memory=True if torch.cuda.is_available() else False)
    
    # Modelo
    model = UNetRefinement(
        n_channels=1, 
        n_classes=1, 
        base_filters=base_filters,
        bilinear=True
    ).to(device)
    
    # Otimizador e scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                     patience=5, verbose=True)
    
    # Loss function
    criterion = PerceptualLossDEM(alpha_mse=0.5, alpha_grad=0.3, alpha_ssim=0.2).to(device)
    
    # Configura logger
    config = {
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'patch_size': patch_size,
        'base_filters': base_filters,
    }
    logger.set_config(config)
    
    # Conta parâmetros
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parâmetros do modelo: {total_params:,}")
    print(f"Log será salvo em: {logger.get_log_file()}\n")
    
    # Treinamento
    model.train()
    best_loss = float('inf')
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_mse = 0
        epoch_grad = 0
        epoch_ssim = 0
        num_batches = 0
        
        for batch_idx, (low_quality, high_quality) in enumerate(dataloader):
            low_quality = low_quality.to(device)
            high_quality = high_quality.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(low_quality)
            
            # Loss
            loss, mse_loss, grad_loss, ssim_loss = criterion(predictions, high_quality)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Acumula métricas
            epoch_loss += loss.item()
            epoch_mse += mse_loss.item()
            epoch_grad += grad_loss.item()
            epoch_ssim += ssim_loss.item()
            num_batches += 1
            
            # Log batch
            lr = optimizer.param_groups[0]['lr']
            logger.log_batch(epoch+1, batch_idx, loss.item(), 
                           mse_loss.item(), grad_loss.item(), ssim_loss.item(), lr)
            
            if batch_idx % 50 == 0:
                print(f"Época [{epoch+1}/{epochs}], Batch [{batch_idx}], "
                      f"Loss: {loss.item():.6f}, LR: {lr:.6e}")
        
        # Médias da época
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
        avg_mse = epoch_mse / num_batches
        avg_grad = epoch_grad / num_batches
        avg_ssim = epoch_ssim / num_batches
        
        # Scheduler
        scheduler.step(avg_epoch_loss)
        lr = optimizer.param_groups[0]['lr']
        
        # Calcula PSNR médio
        epoch_psnr = 0
        psnr_count = 0
        model.eval()
        with torch.no_grad():
            for low_val, high_val in dataloader:
                low_val = low_val.to(device)
                high_val = high_val.to(device)
                pred_val = model(low_val)
                psnr = calculate_psnr(pred_val, high_val)
                epoch_psnr += psnr
                psnr_count += 1
                if psnr_count >= 10:  # Apenas 10 batches para validação
                    break
        model.train()
        
        avg_psnr = epoch_psnr / psnr_count if psnr_count > 0 else 0
        
        # Log época
        logger.log_epoch(epoch+1, avg_epoch_loss, avg_mse, avg_grad, avg_ssim, lr, avg_psnr)
        
        print(f"Época [{epoch+1}/{epochs}] - Loss: {avg_epoch_loss:.6f}, PSNR: {avg_psnr:.2f}dB")
        
        # Salva melhor modelo
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'base_filters': base_filters,
            }, save_path)
            print(f"✓ Modelo salvo com loss: {best_loss:.6f}")
    
    elapsed = time.time() - start_time
    print(f"\nTempo total de treinamento: {elapsed/60:.2f} minutos")
    print(f"Log salvo em: {logger.get_log_file()}\n")
    
    return model, logger.get_log_file()


# ============================================================================
# INFERÊNCIA
# ============================================================================

def generate_refined_dem(
    model_path, input_raster_path, output_path,
    device=None, tile_size=256, overlap=0.25
):
    """
    Gera DEM refinado usando modelo treinado
    Entrada e saída têm a MESMA resolução espacial
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Carrega modelo
    checkpoint = torch.load(model_path, map_location=device)
    base_filters = checkpoint.get('base_filters', 64)
    
    model = UNetRefinement(
        n_channels=1, 
        n_classes=1, 
        base_filters=base_filters,
        bilinear=True
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Modelo carregado - Base filters: {base_filters}")

    with rasterio.open(input_raster_path) as src:
        img = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata
        
        print(f"Dimensões entrada: {img.shape}")
        print(f"Resolução: {abs(src.transform.a):.2f}m")

        # Cria máscara de dados válidos ANTES de normalizar
        valid_mask_original = np.isfinite(img)
        if nodata is not None:
            valid_mask_original &= (img != nodata)
        
        if not np.any(valid_mask_original):
            print("ERRO: Imagem sem dados válidos!")
            return
        
        valid_data = img[valid_mask_original]
        p2, p98 = np.percentile(valid_data, [2, 98])
        
        print(f"Percentis globais: p2={p2:.2f}, p98={p98:.2f}")
        print(f"Range entrada: [{valid_data.min():.2f}, {valid_data.max():.2f}]")
        print(f"Pixels válidos: {valid_mask_original.sum():,} / {img.size:,}")
        
        # Normaliza APENAS pixels válidos
        img_norm = np.full_like(img, 0.0, dtype=np.float32)  # Preenche com 0
        if p98 - p2 > 0:
            img_norm[valid_mask_original] = np.clip((img[valid_mask_original] - p2) / (p98 - p2), 0, 1)
        else:
            print("AVISO: Range muito pequeno, usando valores originais")
            img_norm[valid_mask_original] = 0.5

        # Output array (MESMAS DIMENSÕES)
        out_height, out_width = img.shape
        output_array = np.full((out_height, out_width), np.nan, dtype=np.float32)  # Inicia com NaN
        weight_array = np.zeros_like(output_array)

        # Janela de ponderação (Hann)
        hann_window = np.outer(np.hanning(tile_size), np.hanning(tile_size))

        step = int(tile_size * (1 - overlap))
        
        n_tiles_h = int(np.ceil((img.shape[0] - tile_size) / step)) + 1 if img.shape[0] > tile_size else 1
        n_tiles_w = int(np.ceil((img.shape[1] - tile_size) / step)) + 1 if img.shape[1] > tile_size else 1
        total_tiles = n_tiles_h * n_tiles_w
        
        processed = 0

        print(f"Processando {total_tiles} tiles ({n_tiles_h}x{n_tiles_w}) com overlap={overlap}")
        print(f"Tile size: {tile_size}, Step: {step}\n")
        
        # Processa tiles
        for i in range(0, img.shape[0], step):
            for j in range(0, img.shape[1], step):
                i_end = min(i + tile_size, img.shape[0])
                j_end = min(j + tile_size, img.shape[1])
                
                tile = img_norm[i:i_end, j:j_end]
                tile_valid = valid_mask_original[i:i_end, j:j_end]  # Máscara de validade
                tile_h, tile_w = tile.shape
                
                # Pula tiles completamente vazios
                if not np.any(tile_valid):
                    continue
                
                # Padding se necessário
                tile_padded = np.zeros((tile_size, tile_size), dtype=np.float32)
                tile_padded[:tile_h, :tile_w] = tile
                
                # Máscara de ponderação APENAS em pixels válidos
                weight_mask = np.zeros((tile_size, tile_size), dtype=np.float32)
                weight_mask[:tile_h, :tile_w][tile_valid] = hann_window[:tile_h, :tile_w][tile_valid]
                
                # Inferência
                tile_tensor = torch.from_numpy(tile_padded).unsqueeze(0).unsqueeze(0).to(device).float()
                
                with torch.no_grad():
                    tile_out = model(tile_tensor)
                    tile_out_np = tile_out.cpu().numpy().squeeze()
                
                # Extrai região válida
                tile_out_valid = tile_out_np[:tile_h, :tile_w]
                weights_valid = weight_mask[:tile_h, :tile_w]
                
                # Acumula APENAS onde tinha dados válidos
                valid_indices = tile_valid
                if np.any(valid_indices):
                    # Inicializa com 0 se for a primeira vez
                    current_slice = output_array[i:i_end, j:j_end]
                    current_slice[np.isnan(current_slice)] = 0
                    output_array[i:i_end, j:j_end] = current_slice
                    
                    output_array[i:i_end, j:j_end][valid_indices] += (
                        tile_out_valid[valid_indices] * weights_valid[valid_indices]
                    )
                    weight_array[i:i_end, j:j_end][valid_indices] += weights_valid[valid_indices]
                
                processed += 1
                if processed % 10 == 0:
                    print(f"Progresso: {processed}/{total_tiles} tiles ({processed/total_tiles*100:.1f}%)")

        print(f"\nTotal processado: {processed} tiles")

        # Normalização por peso
        valid_weights = weight_array > 1e-6
        print(f"Pixels com peso válido: {valid_weights.sum()} / {output_array.size}")
        
        if not np.any(valid_weights):
            print("ERRO: Nenhum pixel com peso válido!")
            return
        
        output_array[valid_weights] /= weight_array[valid_weights]

        # Desnormaliza
        output_array = output_array * (p98 - p2 + 1e-8) + p2

        print(f"Output range final: [{output_array[valid_weights].min():.2f}, {output_array[valid_weights].max():.2f}]")

        # Atualiza perfil (MESMA RESOLUÇÃO)
        profile.update({
            'dtype': 'float32'
        })
        
        if nodata is not None:
            output_array[~valid_weights] = float(nodata)

        # Salva
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(output_array.astype('float32'), 1)

        print(f"\nDEM refinado gerado: {output_path}")
        print(f"   Dimensões: {output_array.shape}")
        print(f"   Resolução: {abs(src.transform.a):.2f}m (mantida)\n")


def generate_refined_dem_robust(
    model_path, input_raster_path, output_path,
    device=None, tile_size=256, overlap=0.25
):
    """
    Versão ultra robusta - preserva EXATAMENTE os pixels válidos da entrada
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Carrega modelo
    print("Carregando modelo...")
    checkpoint = torch.load(model_path, map_location=device)
    base_filters = checkpoint.get('base_filters', 64)
    
    model = UNetRefinement(
        n_channels=1, 
        n_classes=1, 
        base_filters=base_filters,
        bilinear=True
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✓ Modelo carregado (base_filters={base_filters})")

    with rasterio.open(input_raster_path) as src:
        img = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata
        transform = src.transform
        
        print(f"\n=== IMAGEM DE ENTRADA ===")
        print(f"Dimensões: {img.shape}")
        print(f"Resolução: {abs(transform.a):.2f}m")
        print(f"Nodata definido: {nodata}")
        
        # PASSO 1: Identifica pixels VÁLIDOS da entrada
        print("\nIdentificando pixels válidos...")
        valid_mask = np.ones(img.shape, dtype=bool)
        
        # Remove NaN e Inf
        valid_mask &= np.isfinite(img)
        
        # Remove nodata (testa vários formatos comuns)
        if nodata is not None:
            valid_mask &= (img != nodata)
            valid_mask &= (img != float(nodata))
        
        # Remove valores extremos comuns de nodata
        valid_mask &= (img != -9999)
        valid_mask &= (img != -32768)
        valid_mask &= (img != -3.40282e+38)
        
        n_valid = valid_mask.sum()
        n_total = img.size
        print(f"Pixels válidos: {n_valid:,} / {n_total:,} ({n_valid/n_total*100:.1f}%)")
        
        if n_valid == 0:
            print("ERRO: Nenhum pixel válido encontrado!")
            return
        
        # PASSO 2: Extrai apenas dados válidos para calcular estatísticas
        valid_data = img[valid_mask]
        p2, p98 = np.percentile(valid_data, [2, 98])
        vmin, vmax = valid_data.min(), valid_data.max()
        
        print(f"\nEstatísticas dos pixels válidos:")
        print(f"  Range: [{vmin:.2f}, {vmax:.2f}]")
        print(f"  P2-P98: [{p2:.2f}, {p98:.2f}]")
        
        # PASSO 3: Normaliza APENAS pixels válidos
        print("\nNormalizando...")
        img_norm = np.zeros_like(img, dtype=np.float32)
        
        if p98 > p2:
            img_norm[valid_mask] = np.clip(
                (img[valid_mask] - p2) / (p98 - p2), 
                0, 1
            )
        else:
            print("AVISO: P98 == P2, usando normalização min-max")
            if vmax > vmin:
                img_norm[valid_mask] = (img[valid_mask] - vmin) / (vmax - vmin)
            else:
                img_norm[valid_mask] = 0.5
        
        print(f"Normalizado: [{img_norm[valid_mask].min():.3f}, {img_norm[valid_mask].max():.3f}]")
        
        # PASSO 4: Inicializa arrays de saída
        print("\nInicializando output...")
        output_array = np.zeros_like(img, dtype=np.float32)
        weight_array = np.zeros_like(img, dtype=np.float32)
        
        # PASSO 5: Processa em tiles
        step = int(tile_size * (1 - overlap))
        n_tiles_h = max(1, (img.shape[0] - 1) // step + 1)
        n_tiles_w = max(1, (img.shape[1] - 1) // step + 1)
        total_tiles = n_tiles_h * n_tiles_w
        
        print(f"\nProcessando {total_tiles} tiles:")
        print(f"  Tile size: {tile_size}x{tile_size}")
        print(f"  Step: {step}")
        print(f"  Overlap: {overlap*100:.0f}%\n")
        
        # Janela Hann para suavização
        hann = np.outer(np.hanning(tile_size), np.hanning(tile_size)).astype(np.float32)
        
        processed = 0
        skipped = 0
        
        for i in range(0, img.shape[0], step):
            for j in range(0, img.shape[1], step):
                # Extrai tile
                i_end = min(i + tile_size, img.shape[0])
                j_end = min(j + tile_size, img.shape[1])
                
                tile_norm = img_norm[i:i_end, j:j_end]
                tile_valid = valid_mask[i:i_end, j:j_end]
                th, tw = tile_norm.shape
                
                # Pula se não tem dados válidos
                if not tile_valid.any():
                    skipped += 1
                    continue
                
                # Padding
                tile_padded = np.zeros((tile_size, tile_size), dtype=np.float32)
                tile_padded[:th, :tw] = tile_norm
                
                # Inferência
                with torch.no_grad():
                    tile_tensor = torch.from_numpy(tile_padded).unsqueeze(0).unsqueeze(0).to(device)
                    tile_pred = model(tile_tensor)
                    tile_pred_np = tile_pred.cpu().numpy().squeeze()[:th, :tw]
                
                # Pesos (Hann APENAS em pixels válidos)
                weights = np.zeros((th, tw), dtype=np.float32)
                weights[tile_valid] = hann[:th, :tw][tile_valid]
                
                # Acumula APENAS pixels válidos
                output_array[i:i_end, j:j_end][tile_valid] += tile_pred_np[tile_valid] * weights[tile_valid]
                weight_array[i:i_end, j:j_end][tile_valid] += weights[tile_valid]
                
                processed += 1
                if processed % 20 == 0:
                    print(f"  [{processed}/{total_tiles}] tiles processados ({processed/total_tiles*100:.0f}%)")
        
        print(f"\n✓ Processamento completo: {processed} tiles processados, {skipped} pulados")
        
        # PASSO 6: Normaliza por pesos
        print("\nNormalizando por pesos...")
        valid_output = (weight_array > 1e-6) & valid_mask
        n_output_valid = valid_output.sum()
        
        print(f"Pixels com peso > 0: {n_output_valid:,} / {n_valid:,} esperados")
        
        if n_output_valid == 0:
            print("ERRO: Nenhum pixel foi processado com sucesso!")
            return
        
        output_array[valid_output] /= weight_array[valid_output]
        
        # PASSO 7: Desnormaliza
        print("Desnormalizando...")
        if p98 > p2:
            output_array[valid_output] = output_array[valid_output] * (p98 - p2) + p2
        else:
            output_array[valid_output] = output_array[valid_output] * (vmax - vmin) + vmin
        
        print(f"Output range: [{output_array[valid_output].min():.2f}, {output_array[valid_output].max():.2f}]")
        
        # PASSO 8: Restaura nodata EXATAMENTE onde estava
        final_nodata = nodata if nodata is not None else -9999
        output_array[~valid_mask] = final_nodata
        
        print(f"\nNodata restaurado: {(~valid_mask).sum():,} pixels")
        
        # PASSO 9: Salva
        print("\nSalvando...")
        profile.update({
            'dtype': 'float32',
            'nodata': final_nodata
        })
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(output_array.astype('float32'), 1)
        
        print(f"\n{'='*60}")
        print(f"✓ DEM REFINADO SALVO: {output_path}")
        print(f"{'='*60}")
        print(f"Dimensões: {output_array.shape}")
        print(f"Pixels válidos: {valid_output.sum():,} ({valid_output.sum()/img.size*100:.1f}%)")
        print(f"Range: [{output_array[valid_output].min():.2f}, {output_array[valid_output].max():.2f}]")
        print(f"Nodata: {final_nodata}")
        print(f"{'='*60}\n")


# # Teste rápido standalone
# if __name__ == "__main__":
#     generate_refined_dem_robust(
#         "model/unet_refinement_5m.pth",
#         "dados/ANADEM_Recorte_IPT.tif",
#         "output/ANADEM_Recorte_REFINADO_V2.tif",
#         tile_size=256,
#         overlap=0.25
#     )


# # ============================================================================
# # MAIN
# # ============================================================================

if __name__ == "__main__":
    # Arquivos de entrada (MESMA RESOLUÇÃO)
    low_quality_files = ["dados/anadem_5m.tif"]     # Baixa qualidade
    high_quality_files = ["dados/geosampa_5m.tif"]  # Alta qualidade (ground truth)
    
    print("="*60)
    print("TREINAMENTO U-NET - REFINAMENTO DE DEM")
    print("="*60 + "\n")
    
    # Treinamento
    model, log_file = train_unet_refinement(
        low_quality_files, high_quality_files, 
        epochs=4,
        batch_size=4,
        learning_rate=1e-4,
        patch_size=128,
        base_filters=64,
        save_path="model/unet_refinement_5m.pth"
    )
    
    print("="*60)
    print("GERANDO GRÁFICOS DE TREINAMENTO")
    print("="*60 + "\n")
    
    # Visualização
    TrainingVisualizer.print_summary(log_file)
    TrainingVisualizer.plot_training_results(log_file)
    
    print("="*60)
    print("GERANDO DEM REFINADO")
    print("="*60 + "\n")
    
    # Inferência
    generate_refined_dem(
        "model/unet_refinement_5m.pth",
        "dados/ANADEM_Recorte_IPT_5m.tif",
        "output/ANADEM_Recorte_REFINADO_10ep.tif",
        tile_size=256,
        overlap=0.25
    )
    
    print("="*60)
    print("PROCESSO COMPLETO!")
    print("="*60)