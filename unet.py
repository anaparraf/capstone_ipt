"""
U-Net simétrica SIMPLIFICADA - SEM MÁSCARAS
Versão limpa que não usa máscaras, apenas normalização global consistente
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
from scipy.ndimage import gaussian_filter

# ---------------------------
# Reprodutibilidade e dispositivo
# ---------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# ---------------------------
# Utils: PSNR simples (sem máscara)
# ---------------------------
def calculate_psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return 100.0
    return (20 * torch.log10(torch.tensor(max_val) / torch.sqrt(mse))).item()

# ---------------------------
# Logger & Visualizer
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
            'batches': [],
            'metrics': {
                'train_loss': [],
                'val_loss': [],
                'train_psnr': [],
                'val_psnr': [],
                'learning_rate': []
            }
        }
        print(f"Logger inicializado: {self.log_file}")

    def set_config(self, config):
        self.data['config'] = config

    def log_batch(self, epoch, batch_idx, loss, lr):
        self.data['batches'].append({'epoch': epoch, 'batch': batch_idx, 'loss': float(loss), 'lr': float(lr)})
        self.save()

    def log_epoch(self, epoch, train_loss, val_loss, train_psnr, val_psnr, lr):
        self.data['epochs'].append({
            'epoch': epoch,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss) if val_loss is not None else None,
            'train_psnr': float(train_psnr) if train_psnr is not None else None,
            'val_psnr': float(val_psnr) if val_psnr is not None else None,
            'lr': float(lr)
        })
        self.data['metrics']['train_loss'].append(float(train_loss))
        self.data['metrics']['val_loss'].append(float(val_loss) if val_loss is not None else float('nan'))
        self.data['metrics']['train_psnr'].append(float(train_psnr) if train_psnr is not None else float('nan'))
        self.data['metrics']['val_psnr'].append(float(val_psnr) if val_psnr is not None else float('nan'))
        self.data['metrics']['learning_rate'].append(float(lr))
        self.save()

    def save(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.data, f, indent=2)

    def get_log_file(self):
        return str(self.log_file)

class TrainingVisualizer:
    @staticmethod
    def load_log(log_file):
        with open(log_file, 'r') as f:
            return json.load(f)

    @staticmethod
    def plot_training_curves(log_file, out_png=None):
        data = TrainingVisualizer.load_log(log_file)
        metrics = data['metrics']
        epochs = np.arange(1, len(metrics['train_loss']) + 1)

        fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(2, 1, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(epochs, metrics['train_loss'], label='train_loss')
        ax1.plot(epochs, metrics['val_loss'], label='val_loss')
        ax1.set_xlabel('Época'); ax1.set_ylabel('Loss'); ax1.legend(); ax1.grid(True)

        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(epochs, metrics['train_psnr'], label='train_psnr')
        ax2.plot(epochs, metrics['val_psnr'], label='val_psnr')
        ax2.set_xlabel('Época'); ax2.set_ylabel('PSNR (dB)'); ax2.legend(); ax2.grid(True)

        plt.suptitle('Treinamento - Loss e PSNR')
        if out_png is None:
            out_png = log_file.replace('.json', '_curves.png')
        plt.savefig(out_png, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Curvas salvas em: {out_png}")
        return out_png

# ---------------------------
# Dataset SIMPLES - SEM MÁSCARAS
# ---------------------------
class SimpleDataset(Dataset):
    def __init__(self, low_res_files, high_res_files, patch_size=128, samples_per_file=2000, 
                 global_norm_stats=None):
        assert len(low_res_files) == len(high_res_files)
        self.low_res = [rasterio.open(p) for p in low_res_files]
        self.high_res = [rasterio.open(p) for p in high_res_files]
        self.patch_size = patch_size
        self.samples_per_file = samples_per_file
        
        # Calcular estatísticas globais
        if global_norm_stats is None:
            print("Calculando estatísticas globais...")
            self.global_norm_stats = self._compute_global_stats()
        else:
            self.global_norm_stats = global_norm_stats
        
        print(f"Estatísticas globais: {self.global_norm_stats}")

    def _compute_global_stats(self):
        all_low_samples = []
        all_high_samples = []
        
        for low_src, high_src in zip(self.low_res, self.high_res):
            low_data = low_src.read(1).astype(np.float32)
            high_data = high_src.read(1).astype(np.float32)
            
            # Remover nodata e infinitos
            low_valid = low_data[np.isfinite(low_data)]
            high_valid = high_data[np.isfinite(high_data)]
            
            if low_src.nodata is not None:
                low_valid = low_valid[low_valid != low_src.nodata]
            if high_src.nodata is not None:
                high_valid = high_valid[high_valid != high_src.nodata]
            
            # Amostragem
            if len(low_valid) > 100000:
                low_valid = np.random.choice(low_valid, 100000, replace=False)
            if len(high_valid) > 100000:
                high_valid = np.random.choice(high_valid, 100000, replace=False)
            
            all_low_samples.append(low_valid)
            all_high_samples.append(high_valid)
        
        all_low = np.concatenate(all_low_samples)
        all_high = np.concatenate(all_high_samples)
        
        return {
            'low_p2': float(np.percentile(all_low, 2)),
            'low_p98': float(np.percentile(all_low, 98)),
            'high_p2': float(np.percentile(all_high, 2)),
            'high_p98': float(np.percentile(all_high, 98))
        }

    def __len__(self):
        return len(self.low_res) * self.samples_per_file

    def _normalize(self, img, nodata, is_low_res=True):
        """Normalização simples sem máscaras"""
        # Remover nodata
        if nodata is not None:
            img = np.where(img == nodata, np.nan, img)
        
        # Substituir NaN/Inf por valor médio
        if np.any(~np.isfinite(img)):
            valid = img[np.isfinite(img)]
            if len(valid) > 0:
                fill_value = np.median(valid)
            else:
                fill_value = 0.0
            img = np.where(np.isfinite(img), img, fill_value)
        
        # Normalizar com estatísticas globais
        if is_low_res:
            p2 = self.global_norm_stats['low_p2']
            p98 = self.global_norm_stats['low_p98']
        else:
            p2 = self.global_norm_stats['high_p2']
            p98 = self.global_norm_stats['high_p98']
        
        eps = 1e-6
        if (p98 - p2) <= eps:
            p98 = p2 + eps
        
        normalized = np.clip((img - p2) / (p98 - p2), 0, 1).astype(np.float32)
        
        # Augmentation leve (20% das vezes)
        if np.random.rand() > 0.8:
            noise = np.random.normal(0, 0.003, normalized.shape).astype(np.float32)
            normalized = np.clip(normalized + noise, 0, 1)
        
        return normalized

    def __getitem__(self, idx):
        file_idx = idx % len(self.low_res)
        low_src = self.low_res[file_idx]
        high_src = self.high_res[file_idx]

        # Garantir que a imagem é grande o suficiente
        if low_src.width < self.patch_size or low_src.height < self.patch_size:
            # Imagem muito pequena, ler tudo e fazer padding
            low = low_src.read(1).astype(np.float32)
            high = high_src.read(1).astype(np.float32)
        else:
            # Extração de patch aleatório
            max_x = low_src.width - self.patch_size
            max_y = low_src.height - self.patch_size
            
            rx = np.random.randint(0, max_x + 1) if max_x > 0 else 0
            ry = np.random.randint(0, max_y + 1) if max_y > 0 else 0
            
            w = Window(rx, ry, self.patch_size, self.patch_size)

            try:
                low = low_src.read(1, window=w).astype(np.float32)
                high = high_src.read(1, window=w).astype(np.float32)
            except:
                low = low_src.read(1).astype(np.float32)
                high = high_src.read(1).astype(np.float32)

        # Normalizar
        low_n = self._normalize(low, low_src.nodata, is_low_res=True)
        high_n = self._normalize(high, high_src.nodata, is_low_res=False)

        # GARANTIR TAMANHO EXATO (patch_size x patch_size)
        h, w = low_n.shape
        
        # Se menor que patch_size, fazer padding
        if h < self.patch_size or w < self.patch_size:
            low_padded = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
            high_padded = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
            low_padded[:h, :w] = low_n
            high_padded[:h, :w] = high_n
            low_n = low_padded
            high_n = high_padded
        
        # Se maior que patch_size, fazer crop
        elif h > self.patch_size or w > self.patch_size:
            low_n = low_n[:self.patch_size, :self.patch_size]
            high_n = high_n[:self.patch_size, :self.patch_size]
        
        # Garantir que high_n tem o mesmo tamanho que low_n
        if high_n.shape != low_n.shape:
            high_padded = np.zeros_like(low_n)
            min_h = min(high_n.shape[0], low_n.shape[0])
            min_w = min(high_n.shape[1], low_n.shape[1])
            high_padded[:min_h, :min_w] = high_n[:min_h, :min_w]
            high_n = high_padded

        low_t = torch.from_numpy(low_n).unsqueeze(0)
        high_t = torch.from_numpy(high_n).unsqueeze(0)

        return low_t, high_t

    def __del__(self):
        for d in getattr(self, 'low_res', []) + getattr(self, 'high_res', []):
            try:
                d.close()
            except:
                pass

# ---------------------------
# Model: Symmetric UNet
# ---------------------------
def init_weights_he(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm_type='group', dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
        if norm_type == 'group':
            groups = min(8, out_ch)
            self.norm1 = nn.GroupNorm(groups, out_ch)
            self.norm2 = nn.GroupNorm(groups, out_ch)
        else:
            self.norm1 = nn.BatchNorm2d(out_ch)
            self.norm2 = nn.BatchNorm2d(out_ch)
        
        self.need_proj = (in_ch != out_ch)
        if self.need_proj:
            self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if self.need_proj:
            identity = self.proj(identity)
        out = out + identity
        out = self.relu(out)
        return out

class SymmetricUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_filters=32, depth=4, norm_type='group', dropout=0.0):
        super().__init__()
        self.depth = depth
        self.enc_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        for i in range(depth):
            in_ch = in_channels if i == 0 else base_filters * (2 ** (i - 1))
            out_ch = base_filters * (2 ** i)
            self.enc_blocks.append(ResidualConvBlock(in_ch, out_ch, norm_type=norm_type, dropout=dropout))

        bottleneck_ch = base_filters * (2 ** (depth - 1))
        self.bottleneck = ResidualConvBlock(bottleneck_ch, bottleneck_ch * 2, norm_type=norm_type, dropout=dropout)

        self.up_convs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in reversed(range(depth)):
            up_in_ch = (bottleneck_ch * 2) if i == (depth - 1) else base_filters * (2 ** (i + 1))
            up_out_ch = base_filters * (2 ** i)
            self.up_convs.append(nn.ConvTranspose2d(up_in_ch, up_out_ch, kernel_size=2, stride=2))
            self.dec_blocks.append(ResidualConvBlock(up_out_ch + up_out_ch, up_out_ch, norm_type=norm_type, dropout=dropout))

        self.final_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)
        self.apply(init_weights_he)

    def forward(self, x):
        skips = []
        out = x
        for enc in self.enc_blocks:
            out = enc(out)
            skips.append(out)
            out = self.pool(out)

        out = self.bottleneck(out)

        for i, (up, dec) in enumerate(zip(self.up_convs, self.dec_blocks)):
            out = up(out)
            skip = skips[-(i + 1)]
            if out.size()[2:] != skip.size()[2:]:
                out = F.interpolate(out, size=skip.size()[2:], mode='bilinear', align_corners=False)
            out = torch.cat([out, skip], dim=1)
            out = dec(out)

        out = self.final_conv(out)
        return out

# ---------------------------
# Loss Function SIMPLES - SEM MÁSCARAS
# ---------------------------
class SimpleLoss(nn.Module):
    def __init__(self, mse_weight=0.7, l1_weight=0.2, grad_weight=0.1):
        super().__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.grad_weight = grad_weight

    def forward(self, pred, target):
        # MSE
        mse_loss = F.mse_loss(pred, target)
        
        # L1
        l1_loss = F.l1_loss(pred, target)
        
        # Gradiente (preserva estruturas)
        pgx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pgy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        tgx = target[:, :, :, 1:] - target[:, :, :, :-1]
        tgy = target[:, :, 1:, :] - target[:, :, :-1, :]
        
        grad_loss = F.l1_loss(pgx, tgx) + F.l1_loss(pgy, tgy)
        
        total = self.mse_weight * mse_loss + self.l1_weight * l1_loss + self.grad_weight * grad_loss
        return total

# ---------------------------
# Training
# ---------------------------
def train_and_validate(
    train_low_files, train_high_files,
    val_low_files, val_high_files,
    epochs=30, batch_size=2, patch_size=128,
    base_filters=32, depth=4, lr=1e-4, weight_decay=1e-6,
    save_path="model/unet_simple.pth",
    samples_per_file=1000, norm_type='group', dropout=0.03
):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    logger = TrainingLogger(log_dir="logs")

    train_ds = SimpleDataset(train_low_files, train_high_files, patch_size=patch_size, samples_per_file=samples_per_file)
    val_ds = SimpleDataset(val_low_files, val_high_files, patch_size=patch_size, samples_per_file=max(200, samples_per_file//10), 
                          global_norm_stats=train_ds.global_norm_stats)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    model = SymmetricUNet(in_channels=1, out_channels=1, base_filters=base_filters, depth=depth, norm_type=norm_type, dropout=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6, verbose=True)
    criterion = SimpleLoss(mse_weight=0.7, l1_weight=0.2, grad_weight=0.1).to(device)

    config = {
        'epochs': epochs, 'batch_size': batch_size, 'patch_size': patch_size,
        'base_filters': base_filters, 'depth': depth, 'lr': lr, 'weight_decay': weight_decay,
        'norm_type': norm_type, 'dropout': dropout,
        'global_norm_stats': train_ds.global_norm_stats
    }
    logger.set_config(config)

    best_val_loss = float('inf')
    best_checkpoint = None

    print(f"Inicio do treino: {epochs} épocas, lr={lr}")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        n_batches = 0

        for batch_idx, (low, high) in enumerate(train_loader):
            low = low.to(device)
            high = high.to(device)

            optimizer.zero_grad()
            preds = model(low)
            loss = criterion(preds, high)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

            if batch_idx % 50 == 0:
                lr_now = optimizer.param_groups[0]['lr']
                print(f"Epoch [{epoch}/{epochs}] Batch [{batch_idx}] loss={loss.item():.6f} lr={lr_now:.2e}")
                logger.log_batch(epoch, batch_idx, loss.item(), lr_now)

        avg_train_loss = running_loss / n_batches if n_batches > 0 else float('nan')

        # Validation
        model.eval()
        val_loss_acc = 0.0
        val_batches = 0
        val_psnr_acc = 0.0

        with torch.no_grad():
            for low_v, high_v in val_loader:
                low_v = low_v.to(device)
                high_v = high_v.to(device)
                out_v = model(low_v)
                loss_v = criterion(out_v, high_v)
                val_loss_acc += loss_v.item()
                val_batches += 1
                psnr_v = calculate_psnr(out_v, high_v)
                val_psnr_acc += psnr_v

        avg_val_loss = val_loss_acc / val_batches if val_batches > 0 else float('nan')
        avg_val_psnr = val_psnr_acc / val_batches if val_batches > 0 else float('nan')

        scheduler.step(avg_val_loss)

        lr_now = optimizer.param_groups[0]['lr']
        logger.log_epoch(epoch, avg_train_loss, avg_val_loss, None, avg_val_psnr, lr_now)

        elapsed = time.time() - epoch_start
        print(f"Epoch {epoch}: train_loss={avg_train_loss:.6f}, val_loss={avg_val_loss:.6f}, val_psnr={avg_val_psnr:.2f} dB, time={elapsed/60:.2f} min")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'config': config
            }
            torch.save(best_checkpoint, save_path)
            print(f"✓ Melhor modelo salvo: val_loss={best_val_loss:.6f}")

    print("Treinamento finalizado.")
    return model, logger.get_log_file(), best_checkpoint

# ---------------------------
# GERAÇÃO LIMPA - SEM MÁSCARAS
# ---------------------------
def generate_clean(model_ckpt_path, input_raster, output_raster, tile_size=512, overlap=0.25, device=None):
    """Geração limpa sem máscaras e com blending suave"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ckpt = torch.load(model_ckpt_path, map_location=device)
    cfg = ckpt.get('config', {})
    base_filters = cfg.get('base_filters', 32)
    depth = cfg.get('depth', 4)
    norm_type = cfg.get('norm_type', 'group')
    global_norm_stats = cfg.get('global_norm_stats', None)
    
    model = SymmetricUNet(in_channels=1, out_channels=1, base_filters=base_filters, depth=depth, norm_type=norm_type).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print(f"Gerando output com tiles {tile_size}x{tile_size}, overlap={overlap*100:.0f}%")

    with rasterio.open(input_raster) as src:
        img = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata
        
        # Substituir nodata por mediana
        if nodata is not None:
            img = np.where(img == nodata, np.nan, img)
        
        valid = img[np.isfinite(img)]
        if len(valid) == 0:
            print("Imagem sem dados válidos!")
            return
        
        fill_value = np.median(valid)
        img = np.where(np.isfinite(img), img, fill_value)
        
        # Normalização global
        if global_norm_stats is not None:
            p2 = global_norm_stats['low_p2']
            p98 = global_norm_stats['low_p98']
            print(f"Usando stats globais: p2={p2:.2f}, p98={p98:.2f}")
        else:
            p2, p98 = np.percentile(img, [2, 98])
            print(f"Stats locais: p2={p2:.2f}, p98={p98:.2f}")
        
        eps = 1e-6
        if (p98 - p2) <= eps:
            p98 = p2 + eps
        
        img_norm = np.clip((img - p2) / (p98 - p2), 0, 1).astype(np.float32)

        out_h, out_w = img.shape
        output = np.zeros((out_h, out_w), dtype=np.float32)
        weight = np.zeros_like(output, dtype=np.float32)
        
        # Máscara de blending suave
        def create_blend_mask(size, border):
            mask = np.ones((size, size), dtype=np.float32)
            for i in range(border):
                alpha = (i + 1) / border
                mask[i, :] *= alpha
                mask[-(i+1), :] *= alpha
                mask[:, i] *= alpha
                mask[:, -(i+1)] *= alpha
            return mask
        
        border = int(tile_size * overlap / 2)
        blend_mask = create_blend_mask(tile_size, border)
        
        step = int(tile_size * (1 - overlap))
        rows = list(range(0, out_h, step))
        cols = list(range(0, out_w, step))
        
        if rows[-1] + tile_size < out_h:
            rows.append(out_h - tile_size)
        if cols[-1] + tile_size < out_w:
            cols.append(out_w - tile_size)
        
        total = len(rows) * len(cols)
        processed = 0

        print(f"Processando {total} tiles...")
        
        for i in rows:
            for j in cols:
                i_start = max(0, i)
                j_start = max(0, j)
                i_end = min(i_start + tile_size, out_h)
                j_end = min(j_start + tile_size, out_w)
                
                tile = img_norm[i_start:i_end, j_start:j_end]
                th, tw = tile.shape
                
                if th < tile_size or tw < tile_size:
                    pad = np.zeros((tile_size, tile_size), dtype=np.float32)
                    pad[:th, :tw] = tile
                    tile_input = pad
                else:
                    tile_input = tile
                
                t_tensor = torch.from_numpy(tile_input).unsqueeze(0).unsqueeze(0).to(device)
                with torch.no_grad():
                    out_tile = model(t_tensor).cpu().numpy().squeeze()
                
                out_valid = out_tile[:th, :tw]
                weight_mask = blend_mask[:th, :tw]
                
                output[i_start:i_end, j_start:j_end] += out_valid * weight_mask
                weight[i_start:i_end, j_start:j_end] += weight_mask
                
                processed += 1
                if processed % 50 == 0 or processed == total:
                    print(f"  {processed}/{total} ({100*processed/total:.1f}%)")

        # Normalizar
        valid_w = weight > 1e-6
        output[valid_w] /= weight[valid_w]
        
        # Desnormalizar
        output = output * (p98 - p2) + p2
        
        # Aplicar nodata onde necessário
        if nodata is not None:
            output[~valid_w] = float(nodata)
        
        # Suavização MUITO LEVE (opcional - remover se não quiser)
        # output = gaussian_filter(output, sigma=0.3)
        
        profile.update(dtype='float32', count=1, nodata=nodata)
        os.makedirs(os.path.dirname(output_raster) or ".", exist_ok=True)
        
        with rasterio.open(output_raster, 'w', **profile) as dst:
            dst.write(output.astype(np.float32), 1)
        
        print(f"✓ Output salvo: {output_raster}")
        print(f"  Stats: min={output[valid_w].min():.2f}, max={output[valid_w].max():.2f}, mean={output[valid_w].mean():.2f}")


# ---------------------------
# Execução principal
# ---------------------------
if __name__ == "__main__":
    # AJUSTE ESTES CAMINHOS
    train_low_files = ["dados/anadem_5m.tif"]
    train_high_files = ["dados/geosampa_5m_reprojetado.tif"]
    val_low_files = ["dados/ANADEM_Recorte_IPT_5m.tif"]
    val_high_files = ["dados/GEOSAMPA_Recorte_IPT_reamostrado_5m.tif"]

    os.makedirs("model", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    # TREINAMENTO
    print("="*60)
    print("INICIANDO TREINAMENTO (SEM MÁSCARAS)")
    print("="*60)
    
    model, log_file, best_ckpt = train_and_validate(
        train_low_files, train_high_files,
        val_low_files, val_high_files,
        epochs=40,              # Reduzi para testar mais rápido
        batch_size=2,           # Voltei para 2 (mais seguro)
        patch_size=128,
        base_filters=32,
        depth=4,
        lr=1e-4,
        weight_decay=1e-6,
        save_path="model/unet_clean_nomask.pth",
        samples_per_file=500,   # Reduzi para treinar mais rápido
        norm_type='group',
        dropout=0.02
    )

    # GRÁFICOS
    print("="*60)
    print("GERANDO GRÁFICOS")
    print("="*60)
    TrainingVisualizer.plot_training_curves(log_file)

    # INFERÊNCIA LIMPA
    print("="*60)
    print("INFERÊNCIA LIMPA (SEM MÁSCARAS)")
    print("="*60)
    
    ckpt_path = "model/unet_clean_nomask.pth"
    if os.path.exists(ckpt_path):
        generate_clean(
            ckpt_path, 
            "dados/ANADEM_Recorte_IPT_5m.tif", 
            "output/ANADEM_clean_nomask.tif", 
            tile_size=1024,
            overlap=0.6,
            device=device
        )
    else:
        print(f"Checkpoint não encontrado: {ckpt_path}")
    
    print("="*60)
    print("FINALIZADO!")
    print("="*60)