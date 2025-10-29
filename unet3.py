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


def calculate_psnr(pred, target, max_val=1.0):
    """Calcula PSNR entre predição e target"""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return 100.0
    psnr = 20 * torch.log10(torch.tensor(max_val) / torch.sqrt(mse))
    return psnr.item()


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
                'l1_loss': [],
                'grad_loss': [],
                'ssim_loss': [],
                'learning_rate': [],
                'psnr': [],
            }
        }
        
        print(f"Logger inicializado: {self.log_file}")
    
    def set_config(self, config_dict):
        self.data['config'] = config_dict
    
    def log_batch(self, epoch, batch_idx, loss, mse, l1, grad, ssim, lr):
        self.data['batches'].append({
            'epoch': epoch,
            'batch': batch_idx,
            'loss': float(loss),
            'mse': float(mse),
            'l1': float(l1),
            'grad': float(grad),
            'ssim': float(ssim),
            'lr': float(lr),
        })
    
    def log_epoch(self, epoch, epoch_loss, mse, l1, grad, ssim, lr, psnr=None):
        self.data['epochs'].append({
            'epoch': epoch,
            'loss': float(epoch_loss),
            'mse': float(mse),
            'l1': float(l1),
            'grad': float(grad),
            'ssim': float(ssim),
            'lr': float(lr),
            'psnr': float(psnr) if psnr is not None else None,
        })
        
        self.data['metrics']['epoch_loss'].append(float(epoch_loss))
        self.data['metrics']['mse_loss'].append(float(mse))
        self.data['metrics']['l1_loss'].append(float(l1))
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
        
        fig = plt.figure(figsize=(18, 14))
        gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(epochs, metrics['epoch_loss'], 'b-', linewidth=2, label='Total Loss')
        ax1.fill_between(epochs, metrics['epoch_loss'], alpha=0.3)
        ax1.set_xlabel('Época', fontsize=11)
        ax1.set_ylabel('Loss', fontsize=11)
        ax1.set_title('Loss Total por Época', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(epochs, metrics['mse_loss'], 'g-', linewidth=2, label='MSE Loss')
        ax2.fill_between(epochs, metrics['mse_loss'], alpha=0.3, color='green')
        ax2.set_xlabel('Época', fontsize=11)
        ax2.set_ylabel('MSE Loss', fontsize=11)
        ax2.set_title('Mean Squared Error', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(epochs, metrics['l1_loss'], 'orange', linewidth=2, label='L1 Loss')
        ax3.fill_between(epochs, metrics['l1_loss'], alpha=0.3, color='orange')
        ax3.set_xlabel('Época', fontsize=11)
        ax3.set_ylabel('L1 Loss', fontsize=11)
        ax3.set_title('L1 (Mean Absolute Error)', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(epochs, metrics['grad_loss'], 'r-', linewidth=2, label='Gradient Loss')
        ax4.fill_between(epochs, metrics['grad_loss'], alpha=0.3, color='red')
        ax4.set_xlabel('Época', fontsize=11)
        ax4.set_ylabel('Gradient Loss', fontsize=11)
        ax4.set_title('Preservação de Bordas', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(epochs, metrics['ssim_loss'], 'purple', linewidth=2, label='SSIM Loss')
        ax5.fill_between(epochs, metrics['ssim_loss'], alpha=0.3, color='purple')
        ax5.set_xlabel('Época', fontsize=11)
        ax5.set_ylabel('SSIM Loss', fontsize=11)
        ax5.set_title('Similaridade Estrutural', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.plot(epochs, metrics['learning_rate'], 'brown', linewidth=2, label='Learning Rate')
        ax6.fill_between(epochs, metrics['learning_rate'], alpha=0.3, color='brown')
        ax6.set_xlabel('Época', fontsize=11)
        ax6.set_ylabel('Learning Rate', fontsize=11)
        ax6.set_title('Taxa de Aprendizado (Adaptativo)', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        ax6.set_yscale('log')
        ax6.legend()
        
        # PSNR Plot
        if metrics.get('psnr') and len(metrics['psnr']) > 0:
            ax7 = fig.add_subplot(gs[3, :])
            ax7.plot(epochs[:len(metrics['psnr'])], metrics['psnr'], 'darkgreen', linewidth=2.5, marker='o', markersize=5, label='PSNR')
            ax7.fill_between(epochs[:len(metrics['psnr'])], metrics['psnr'], alpha=0.3, color='darkgreen')
            ax7.set_xlabel('Época', fontsize=11)
            ax7.set_ylabel('PSNR (dB)', fontsize=11)
            ax7.set_title('Peak Signal-to-Noise Ratio (PSNR)', fontsize=12, fontweight='bold')
            ax7.grid(True, alpha=0.3)
            ax7.legend()
        
        plt.suptitle('Monitoramento de Treinamento - U-Net Super-Resolução', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        if output_file is None:
            output_file = log_file.replace('.json', '_plots.png')
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Gráficos salvos em: {output_file}")
        plt.show()
        
        return output_file
    
    @staticmethod
    def plot_all_losses_combined(log_file, output_file=None):
        data = TrainingVisualizer.load_log(log_file)
        metrics = data['metrics']
        
        epochs = range(1, len(metrics['epoch_loss']) + 1)
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        mse_norm = np.array(metrics['mse_loss']) / np.max(metrics['mse_loss'])
        l1_norm = np.array(metrics['l1_loss']) / np.max(metrics['l1_loss'])
        grad_norm = np.array(metrics['grad_loss']) / np.max(metrics['grad_loss'])
        ssim_norm = np.array(metrics['ssim_loss']) / np.max(metrics['ssim_loss'])
        
        ax.plot(epochs, mse_norm, 'g-', linewidth=2.5, label='MSE (Normalizado)', marker='o', markersize=4)
        ax.plot(epochs, l1_norm, 'orange', linewidth=2.5, label='L1 (Normalizado)', marker='s', markersize=4)
        ax.plot(epochs, grad_norm, 'r-', linewidth=2.5, label='Gradient (Normalizado)', marker='^', markersize=4)
        ax.plot(epochs, ssim_norm, 'purple', linewidth=2.5, label='SSIM (Normalizado)', marker='d', markersize=4)
        
        ax.set_xlabel('Época', fontsize=12, fontweight='bold')
        ax.set_ylabel('Valor Normalizado [0-1]', fontsize=12, fontweight='bold')
        ax.set_title('Comparação de Todas as Componentes de Loss (Normalizadas)', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        
        if output_file is None:
            output_file = log_file.replace('.json', '_combined_losses.png')
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Gráfico combinado salvo em: {output_file}")
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
        print(f"  L1 Loss:       {metrics['l1_loss'][-1]:.6f} (init: {metrics['l1_loss'][0]:.6f})")
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
        print(f"  Redução: {metrics['learning_rate'][0] / metrics['learning_rate'][-1]:.2f}x")
        print("="*60 + "\n")


class AdaptiveSuperResTiffDataset(Dataset):
    def __init__(self, low_res_files, high_res_files, target_resolution=None, transform=None, patch_size=128):
        self.low_res_datasets = [rasterio.open(f) for f in low_res_files]
        self.high_res_datasets = [rasterio.open(f) for f in high_res_files]
        self.transform = transform
        self.patch_size = patch_size
        self.target_resolution = target_resolution
        
        assert len(self.low_res_datasets) == len(self.high_res_datasets)
        self._calculate_resolution_info()
    
    def _calculate_resolution_info(self):
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
        
        img_low_norm = self._normalize_image(img_low, low_src.nodata)
        img_high_norm = self._normalize_image(img_high, high_src.nodata)
        
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
        
        return low_tensor, high_tensor, scale_factor
    
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
            
            if np.random.random() > 0.5:
                noise = np.random.normal(0, 0.02, img_norm.shape).astype(np.float32)
                img_norm = np.clip(img_norm + noise, 0, 1)
            
            return img_norm
        else:
            return np.zeros_like(img, dtype=np.float32)
    
    def __del__(self):
        for dataset in self.low_res_datasets + self.high_res_datasets:
            if hasattr(dataset, 'close'):
                dataset.close()


class ResolutionAwareUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_filters=32, scale_factor=6):
        super().__init__()
        
        self.scale_factor = scale_factor
        f = base_filters
        
        self.conv1 = self._conv_block(in_channels, f)
        self.conv2 = self._conv_block(f, f*2)
        self.conv3 = self._conv_block(f*2, f*4)
        self.conv4 = self._conv_block(f*4, f*8)
        self.conv5 = self._conv_block(f*8, f*16)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.bottleneck = self._conv_block(f*16, f*32)
        
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
        
        self.subpixel_conv = nn.Sequential(
            nn.Conv2d(f, f*4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(f, f*4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )
        
        self.final_refine = nn.Sequential(
            nn.Conv2d(f, f//2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f//2, out_channels, kernel_size=1)
        )
    
    def _conv_block(self, in_ch, out_ch):
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
        _, _, h, w = x.shape
        
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
        
        b = self.bottleneck(p5)
        
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
        
        # u_upsampled = self.subpixel_conv(u1)
        
        # if target_scale is not None:
        #     target_h = int(h * target_scale)
        #     target_w = int(w * target_scale)
        #     u_upsampled = F.interpolate(u_upsampled, size=(target_h, target_w),
        #                                mode='bilinear', align_corners=False)
        
        # output = self.final_refine(u_upsampled)
        output = self.final_refine(u1)
        return output
    
    def _match_size_and_concat(self, upsampled, skip):
        if upsampled.size() != skip.size():
            upsampled = F.interpolate(upsampled, size=skip.size()[2:], 
                                    mode='bilinear', align_corners=False)
        return torch.cat([upsampled, skip], 1)


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
    
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        l1_loss = self.l1(pred, target)
        grad_loss = self._gradient_loss(pred, target)
        ssim_loss = self._ssim_loss(pred, target)
        
        return 0.4 * mse_loss + 0.2 * l1_loss + 0.2 * grad_loss + 0.2 * (1 - ssim_loss)
    
    def _gradient_loss(self, pred, target):
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
    
    print(f"Iniciando treinamento para resolução alvo: {target_resolution}m")
    start_time = time.time()
    
    logger = TrainingLogger(log_dir="logs")
    
    dataset = AdaptiveSuperResTiffDataset(
        low_res_files, high_res_files,
        target_resolution=target_resolution,
        patch_size=patch_size
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                           num_workers=0, pin_memory=True if torch.cuda.is_available() else False)
    
    scale_factor = dataset.resolution_info[0]['avg_scale']
    
    model = ResolutionAwareUNet(base_filters=32, scale_factor=int(scale_factor)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                     patience=10, verbose=False)
    criterion = PerceptualLoss().to(device)
    
    model.train()
    best_loss = float('inf')
    
    config = {
        'target_resolution': target_resolution,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'patch_size': patch_size,
        'scale_factor': float(scale_factor),
        'base_filters': 32,
    }
    logger.set_config(config)
    
    print(f"Treinando com scale_factor={scale_factor}")
    print(f"Log será salvo em: {logger.get_log_file()}\n")
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_mse = 0
        epoch_l1 = 0
        epoch_grad = 0
        epoch_ssim = 0
        num_batches = 0
        
        for batch_idx, (low_res, high_res, resolution_factors) in enumerate(dataloader):
            low_res = low_res.to(device)
            high_res = high_res.to(device)
            
            optimizer.zero_grad()
            
            avg_resolution_factor = resolution_factors.mean().item()
            predictions = model(low_res, target_scale=avg_resolution_factor)
            
            if predictions.shape != high_res.shape:
                predictions = F.interpolate(predictions, size=high_res.shape[2:],
                                           mode='bilinear', align_corners=False)
            
            mse_loss = criterion.mse(predictions, high_res)
            l1_loss = criterion.l1(predictions, high_res)
            grad_loss = criterion._gradient_loss(predictions, high_res)
            ssim_loss = criterion._ssim_loss(predictions, high_res)
            
            loss = criterion(predictions, high_res)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_mse += mse_loss.item()
            epoch_l1 += l1_loss.item()
            epoch_grad += grad_loss.item()
            epoch_ssim += ssim_loss.item()
            num_batches += 1
            
            lr = optimizer.param_groups[0]['lr']
            logger.log_batch(epoch+1, batch_idx, loss.item(), 
                           mse_loss.item(), l1_loss.item(), 
                           grad_loss.item(), ssim_loss.item(), lr)
            
            if batch_idx % 50 == 0:
                print(f"Época [{epoch+1}/{epochs}], Batch [{batch_idx}], "
                      f"Loss: {loss.item():.6f}, LR: {lr:.6e}")
        
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
        avg_mse = epoch_mse / num_batches
        avg_l1 = epoch_l1 / num_batches
        avg_grad = epoch_grad / num_batches
        avg_ssim = epoch_ssim / num_batches
        
        scheduler.step(avg_epoch_loss)
        lr = optimizer.param_groups[0]['lr']
        
        # Calcular PSNR médio da época
        epoch_psnr = 0
        psnr_count = 0
        model.eval()
        with torch.no_grad():
            for low_res_val, high_res_val, _ in dataloader:
                low_res_val = low_res_val.to(device)
                high_res_val = high_res_val.to(device)
                predictions_val = model(low_res_val, target_scale=avg_resolution_factor)
                if predictions_val.shape != high_res_val.shape:
                    predictions_val = F.interpolate(predictions_val, size=high_res_val.shape[2:],
                                                   mode='bilinear', align_corners=False)
                psnr = calculate_psnr(predictions_val, high_res_val)
                epoch_psnr += psnr
                psnr_count += 1
        model.train()
        
        avg_psnr = epoch_psnr / psnr_count if psnr_count > 0 else 0
        
        logger.log_epoch(epoch+1, avg_epoch_loss, avg_mse, avg_l1, avg_grad, avg_ssim, lr, avg_psnr)
        
        print(f"Época [{epoch+1}/{epochs}] - Loss média: {avg_epoch_loss:.6f}, PSNR: {avg_psnr:.2f}dB")
        
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
            print(f"Modelo salvo com loss: {best_loss:.6f}")
    
    elapsed = time.time() - start_time
    print(f"\nTempo total de treinamento: {elapsed/60:.2f} minutos")
    print(f"Log salvo em: {logger.get_log_file()}\n")
    
    return model, logger.get_log_file()


def generate_super_resolution(
    model_path, input_raster_path, output_path,
    target_resolution=None, device=None, tile_size=256, overlap=0.5
):
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

        out_height = int(np.ceil(img.shape[0] * calc_scale))
        out_width = int(np.ceil(img.shape[1] * calc_scale))
        
        print(f"Dimensões saída: ({out_height}, {out_width})")
        
        output_array = np.zeros((out_height, out_width), dtype=np.float32)
        weight_array = np.zeros_like(output_array)

        hann_window = np.outer(np.hanning(tile_size), np.hanning(tile_size))

        step = int(tile_size * (1 - overlap))
        
        n_tiles_h = int(np.ceil((img.shape[0] - tile_size) / step)) + 1 if img.shape[0] > tile_size else 1
        n_tiles_w = int(np.ceil((img.shape[1] - tile_size) / step)) + 1 if img.shape[1] > tile_size else 1
        total_tiles = n_tiles_h * n_tiles_w
        
        processed = 0

        print(f"Processando {total_tiles} tiles ({n_tiles_h}x{n_tiles_w}) com overlap={overlap}")
        print(f"Tile size: {tile_size}, Step: {step}\n")
        
        for i in range(0, img.shape[0], step):
            for j in range(0, img.shape[1], step):
                i_end = min(i + tile_size, img.shape[0])
                j_end = min(j + tile_size, img.shape[1])
                
                tile = img_norm[i:i_end, j:j_end]
                tile_h, tile_w = tile.shape
                
                tile_padded = np.zeros((tile_size, tile_size), dtype=np.float32)
                tile_padded[:tile_h, :tile_w] = tile
                
                weight_mask = np.zeros((tile_size, tile_size), dtype=np.float32)
                weight_mask[:tile_h, :tile_w] = hann_window[:tile_h, :tile_w]
                
                tile_tensor = torch.from_numpy(tile_padded).unsqueeze(0).unsqueeze(0).to(device).float()
                
                with torch.no_grad():
                    tile_out = model(tile_tensor, target_scale=calc_scale)
                    tile_out_np = tile_out.cpu().numpy().squeeze()
                
                tile_out_h = int(np.ceil(tile_h * calc_scale))
                tile_out_w = int(np.ceil(tile_w * calc_scale))
                
                tile_out_full_h = int(np.ceil(tile_size * calc_scale))
                tile_out_full_w = int(np.ceil(tile_size * calc_scale))
                
                if tile_out_np.shape != (tile_out_full_h, tile_out_full_w):
                    tile_out_tensor = torch.from_numpy(tile_out_np).unsqueeze(0).unsqueeze(0)
                    tile_out_tensor = F.interpolate(
                        tile_out_tensor, 
                        size=(tile_out_full_h, tile_out_full_w),
                        mode='bilinear', 
                        align_corners=False
                    )
                    tile_out_np = tile_out_tensor.squeeze().cpu().numpy()
                
                tile_out_valid = tile_out_np[:tile_out_h, :tile_out_w]
                
                weights_tensor = torch.from_numpy(weight_mask).unsqueeze(0).unsqueeze(0)
                weights_upsampled = F.interpolate(
                    weights_tensor, 
                    size=(tile_out_full_h, tile_out_full_w),
                    mode='bilinear', 
                    align_corners=False
                )
                weights_valid = weights_upsampled.squeeze().cpu().numpy()[:tile_out_h, :tile_out_w]
                
                out_i = int(np.round(i * calc_scale))
                out_j = int(np.round(j * calc_scale))
                
                out_i_end = min(out_i + tile_out_h, out_height)
                out_j_end = min(out_j + tile_out_w, out_width)
                
                actual_h = out_i_end - out_i
                actual_w = out_j_end - out_j
                
                output_array[out_i:out_i_end, out_j:out_j_end] += tile_out_valid[:actual_h, :actual_w] * weights_valid[:actual_h, :actual_w]
                weight_array[out_i:out_i_end, out_j:out_j_end] += weights_valid[:actual_h, :actual_w]
                
                processed += 1
                if processed % 10 == 0:
                    print(f"Progresso: {processed}/{total_tiles} tiles")

        print(f"\nTotal processado: {processed} tiles")

        valid_weights = weight_array > 1e-6
        print(f"Pixels com peso válido: {valid_weights.sum()} / {output_array.size}")
        
        if not np.any(valid_weights):
            print("ERRO: Nenhum pixel com peso válido!")
            return
        
        output_array[valid_weights] /= weight_array[valid_weights]

        output_array = output_array * (p98 - p2 + 1e-8) + p2

        print(f"Output range final: [{output_array[valid_weights].min():.2f}, {output_array[valid_weights].max():.2f}]")

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
            output_array[~valid_weights] = float(nodata)

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(output_array.astype('float32'), 1)

        print(f"\nSuper resolução gerada: {output_path}")
        print(f"   Resolução final: {target_resolution:.2f}m")
        print(f"   Dimensões finais: {output_array.shape}\n")


if __name__ == "__main__":
    low_res_files = ["dados/anadem_5m.tif"]
    high_res_files = ["dados/geosampa_5m.tif"]
    
    print("="*60)
    print("INICIANDO TREINAMENTO")
    print("="*60 + "\n")
    
    model, log_file = train_adaptive_unet(
        low_res_files, high_res_files, 
        target_resolution=5,
        epochs=20,             
        batch_size=2,
        patch_size=32, #90/5m
        save_path="model/adaptive_unet_5m_samesize.pth"
    )
    
    print("="*60)
    print("GERANDO GRÁFICOS DE TREINAMENTO")
    print("="*60 + "\n")
    
    TrainingVisualizer.print_summary(log_file)
    TrainingVisualizer.plot_training_results(log_file)
    TrainingVisualizer.plot_all_losses_combined(log_file)
    
    print("="*60)
    print("GERANDO SUPER-RESOLUÇÃO")
    print("="*60 + "\n")
    
    generate_super_resolution(
        "model/adaptive_unet_5m_samesize.pth",
        "dados/ANADEM_Recorte_IPT.tif",
        "output/ANADEM_Recorte_teste_20_samesize_32patch.tif",
        target_resolution=5,
        tile_size=256,
        overlap=0.1
    )