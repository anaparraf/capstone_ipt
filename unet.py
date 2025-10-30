"""
U-Net simétrica com Residual Blocks + mask-aware loss + Adam optimizer.
Treina com dataset onde low_res e high_res estão na mesma resolução (ex: 5m -> 5m).
Salva melhores checkpoints por PSNR de validação.

Ajuste caminhos em __main__ antes de rodar.
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
# Utils: PSNR (mask-aware)
# ---------------------------
def calculate_psnr_masked(pred, target, mask, max_val=1.0):
    """
    pred, target: tensors (B,1,H,W)
    mask: same shape with 0/1
    returns average PSNR over batch considering only valid pixels
    """
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
    if len(psnrs) == 0:
        return float('nan')
    return float(np.mean(psnrs))

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
# Dataset (same-resolution, returns mask)
# ---------------------------
class SameResTiffDataset(Dataset):
    def __init__(self, low_res_files, high_res_files, patch_size=128, samples_per_file=2000, transform=None):
        assert len(low_res_files) == len(high_res_files)
        self.low_res = [rasterio.open(p) for p in low_res_files]
        self.high_res = [rasterio.open(p) for p in high_res_files]
        self.patch_size = patch_size
        self.samples_per_file = samples_per_file
        self.transform = transform

        for i, (l, h) in enumerate(zip(self.low_res, self.high_res), start=1):
            lr = (abs(l.transform.a) + abs(l.transform.e)) / 2.0
            hr = (abs(h.transform.a) + abs(h.transform.e)) / 2.0
            print(f"Par {i}: low_res avg={lr:.3f} m, high_res avg={hr:.3f} m")

    def __len__(self):
        return max(1, len(self.low_res) * self.samples_per_file)

    def _normalize_and_mask(self, img, nodata):
        if nodata is not None:
            valid = (img != nodata) & np.isfinite(img)
        else:
            valid = np.isfinite(img)
        valid = valid.astype(np.uint8)
        if not np.any(valid):
            return np.zeros_like(img, dtype=np.float32), valid
        arr = img.copy().astype(np.float32)
        vals = arr[valid == 1]
        p2, p98 = np.percentile(vals, [2, 98])
        eps = 1e-6
        if (p98 - p2) <= eps:
            p98 = p2 + eps
        norm = np.zeros_like(arr, dtype=np.float32)
        norm[valid == 1] = np.clip((arr[valid == 1] - p2) / (p98 - p2), 0, 1)
        # small augmentation noise
        if np.random.rand() > 0.5:
            norm = np.clip(norm + np.random.normal(0, 0.01, norm.shape).astype(np.float32), 0, 1)
        return norm, valid

    def __getitem__(self, idx):
        file_idx = idx % len(self.low_res)
        low_src = self.low_res[file_idx]
        high_src = self.high_res[file_idx]

        max_x = max(0, low_src.width - self.patch_size)
        max_y = max(0, low_src.height - self.patch_size)
        if max_x == 0 or max_y == 0:
            w_low = Window(0, 0, min(low_src.width, self.patch_size), min(low_src.height, self.patch_size))
            w_high = Window(0, 0, min(high_src.width, self.patch_size), min(high_src.height, self.patch_size))
        else:
            rx = np.random.randint(0, max_x + 1)
            ry = np.random.randint(0, max_y + 1)
            w_low = Window(rx, ry, self.patch_size, self.patch_size)
            w_high = Window(rx, ry, self.patch_size, self.patch_size)

        try:
            low = low_src.read(1, window=w_low).astype(np.float32)
            high = high_src.read(1, window=w_high).astype(np.float32)
        except Exception as e:
            # fallback to top-left crop
            low = low_src.read(1).astype(np.float32)[:self.patch_size, :self.patch_size]
            high = high_src.read(1).astype(np.float32)[:self.patch_size, :self.patch_size]

        low_n, mask_low = self._normalize_and_mask(low, low_src.nodata)
        high_n, mask_high = self._normalize_and_mask(high, high_src.nodata)

        # ensure same shape; if mismatch, resample high to low shape
        if low_n.shape != high_n.shape:
            t = torch.from_numpy(high_n).unsqueeze(0).unsqueeze(0)
            t = F.interpolate(t, size=low_n.shape, mode='bilinear', align_corners=False)
            high_n = t.squeeze().numpy()
            mask_high = (F.interpolate(torch.from_numpy(mask_high.astype(np.float32)).unsqueeze(0).unsqueeze(0),
                                       size=low_n.shape, mode='nearest').squeeze().numpy().astype(np.uint8))

        low_t = torch.from_numpy(low_n).unsqueeze(0)   # (1,H,W)
        high_t = torch.from_numpy(high_n).unsqueeze(0)
        mask_t = torch.from_numpy((mask_high > 0).astype(np.float32)).unsqueeze(0)  # prefer mask of target

        if self.transform:
            low_t = self.transform(low_t)
            high_t = self.transform(high_t)

        return low_t, high_t, mask_t

    def __del__(self):
        for d in getattr(self, 'low_res', []) + getattr(self, 'high_res', []):
            try:
                d.close()
            except Exception:
                pass

# ---------------------------
# Model: Symmetric UNet com ResidualConvBlock
# ---------------------------
def init_weights_he(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm_type='group', dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        # norm layers
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
        """
        depth: número de níveis (por exemplo depth=4 => 4 enc blocks + bottleneck + 4 dec blocks)
        base_filters: filtros no primeiro nível (ajustar conforme memória)
        """
        super().__init__()
        self.depth = depth
        self.enc_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        for i in range(depth):
            in_ch = in_channels if i == 0 else base_filters * (2 ** (i - 1))
            out_ch = base_filters * (2 ** i)
            self.enc_blocks.append(ResidualConvBlock(in_ch, out_ch, norm_type=norm_type, dropout=dropout))

        # bottleneck
        bottleneck_ch = base_filters * (2 ** (depth - 1))
        self.bottleneck = ResidualConvBlock(bottleneck_ch, bottleneck_ch * 2, norm_type=norm_type, dropout=dropout)

        # decoder
        self.up_convs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in reversed(range(depth)):
            up_in_ch = (bottleneck_ch * 2) if i == (depth - 1) else base_filters * (2 ** (i + 1))
            up_out_ch = base_filters * (2 ** i)
            self.up_convs.append(nn.ConvTranspose2d(up_in_ch, up_out_ch, kernel_size=2, stride=2))
            # dec block takes concat of up + skip => in_ch = up_out_ch + skip_ch
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
# Mask-aware Perceptual Loss (but can use only mse if desired)
# ---------------------------
class PerceptualLossMasked(nn.Module):
    def __init__(self, eps=1e-6, use_ssim=False):
        super().__init__()
        self.eps = eps
        self.use_ssim = use_ssim

    def forward(self, pred, target, mask):
        """
        pred, target, mask: (B,1,H,W) ; mask values 0/1 float
        Returns scalar loss
        """
        if mask is None:
            mask = torch.ones_like(target, device=target.device)

        # masked mse
        mask_count = torch.clamp(mask.sum(dim=[1,2,3]), min=1.0)  # per-sample count
        mse_map = ((pred - target)**2) * mask
        mse_per_sample = mse_map.view(mse_map.size(0), -1).sum(dim=1) / mask_count
        mse_loss = mse_per_sample.mean()

        # masked l1
        l1_map = torch.abs(pred - target) * mask
        l1_per_sample = l1_map.view(l1_map.size(0), -1).sum(dim=1) / mask_count
        l1_loss = l1_per_sample.mean()

        grad_loss = self._masked_gradient_loss(pred, target, mask)

        if self.use_ssim:
            ssim_val = self._masked_ssim(pred, target, mask)
            ssim_loss = 1.0 - ssim_val
        else:
            ssim_loss = torch.tensor(0.0, device=pred.device)

        # weights: prefer começar com MSE + small L1 + small grad
        total = 0.6 * mse_loss + 0.2 * l1_loss + 0.2 * grad_loss + 0.2 * ssim_loss
        return total

    def _masked_gradient_loss(self, pred, target, mask):
        pgx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pgy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        tgx = target[:, :, :, 1:] - target[:, :, :, :-1]
        tgy = target[:, :, 1:, :] - target[:, :, :-1, :]

        mask_gx = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        mask_gy = mask[:, :, 1:, :] * mask[:, :, :-1, :]

        loss_gx = torch.abs(pgx - tgx) * mask_gx
        loss_gy = torch.abs(pgy - tgy) * mask_gy

        cnt_gx = torch.clamp(mask_gx.sum(dim=[1,2,3]), min=1.0)
        cnt_gy = torch.clamp(mask_gy.sum(dim=[1,2,3]), min=1.0)

        loss_gx_b = loss_gx.view(loss_gx.size(0), -1).sum(dim=1) / cnt_gx
        loss_gy_b = loss_gy.view(loss_gy.size(0), -1).sum(dim=1) / cnt_gy

        return (loss_gx_b + loss_gy_b).mean()

    def _masked_ssim(self, pred, target, mask):
        # robust global approximate SSIM on valid pixels (per sample)
        B = pred.size(0)
        vals = []
        for b in range(B):
            mb = mask[b:b+1].bool()
            if mb.sum() < 16:
                vals.append(torch.tensor(0.0, device=pred.device))
                continue
            p = pred[b:b+1][mb]
            t = target[b:b+1][mb]
            mu_p = p.mean(); mu_t = t.mean()
            sigma_p2 = p.var(unbiased=False)
            sigma_t2 = t.var(unbiased=False)
            sigma_pt = ((p - mu_p) * (t - mu_t)).mean()
            C1 = 0.01 ** 2; C2 = 0.03 ** 2
            denom = (mu_p**2 + mu_t**2 + C1) * (sigma_p2 + sigma_t2 + C2)
            if denom == 0:
                vals.append(torch.tensor(0.0, device=pred.device))
            else:
                ssim_b = ((2*mu_p*mu_t + C1) * (2*sigma_pt + C2)) / denom
                vals.append(torch.clamp(ssim_b, 0.0, 1.0))
        return torch.stack(vals).mean()

# ---------------------------
# Training + Validation loop
# ---------------------------
def train_and_validate(
    train_low_files, train_high_files,
    val_low_files, val_high_files,
    epochs=30, batch_size=2, patch_size=128,
    base_filters=32, depth=4, lr=1e-4, weight_decay=1e-6,
    save_path="model/unet_sym_masked_adam.pth",
    samples_per_file=1000, use_ssim=False, norm_type='group', dropout=0.05
):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    debug_dir = "debug_nan_batches"
    os.makedirs(debug_dir, exist_ok=True)

    logger = TrainingLogger(log_dir="logs")

    train_ds = SameResTiffDataset(train_low_files, train_high_files, patch_size=patch_size, samples_per_file=samples_per_file)
    val_ds = SameResTiffDataset(val_low_files, val_high_files, patch_size=patch_size, samples_per_file=max(200, samples_per_file//10))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    model = SymmetricUNet(in_channels=1, out_channels=1, base_filters=base_filters, depth=depth, norm_type=norm_type, dropout=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6, verbose=True)
    criterion = PerceptualLossMasked(use_ssim=use_ssim).to(device)

    config = {
        'epochs': epochs, 'batch_size': batch_size, 'patch_size': patch_size,
        'base_filters': base_filters, 'depth': depth, 'lr': lr, 'weight_decay': weight_decay,
        'use_ssim': use_ssim, 'norm_type': norm_type, 'dropout': dropout
    }
    logger.set_config(config)

    best_val_psnr = -np.inf
    best_checkpoint = None

    print(f"Inicio do treino: {epochs} épocas, lr={lr}, base_filters={base_filters}, depth={depth}")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        n_batches = 0

        for batch_idx, (low, high, mask) in enumerate(train_loader):
            low = low.to(device); high = high.to(device); mask = mask.to(device)

            # checks
            if not torch.isfinite(low).all() or not torch.isfinite(high).all():
                print(f"[Treino] Epoch {epoch} Batch {batch_idx}: dados inválidos (NaN/Inf) — salvando e pulando")
                fname = os.path.join(debug_dir, f"invalid_train_e{epoch}_b{batch_idx}.npz")
                np.savez_compressed(fname, low=low.cpu().numpy(), high=high.cpu().numpy(), mask=mask.cpu().numpy())
                continue
            if mask.sum() < 1.0:
                continue

            optimizer.zero_grad()
            preds = model(low)

            if not torch.isfinite(preds).all():
                print(f"[Treino] Epoch {epoch} Batch {batch_idx}: preds contém NaN/Inf — salvando e pulando")
                fname = os.path.join(debug_dir, f"prednan_train_e{epoch}_b{batch_idx}.npz")
                np.savez_compressed(fname, low=low.cpu().numpy(), high=high.cpu().numpy(), mask=mask.cpu().numpy(), preds=preds.cpu().detach().numpy())
                continue

            loss = criterion(preds, high, mask)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[Treino] Epoch {epoch} Batch {batch_idx}: loss NaN/Inf — salvando e pulando")
                fname = os.path.join(debug_dir, f"lossnan_train_e{epoch}_b{batch_idx}.npz")
                np.savez_compressed(fname, low=low.cpu().numpy(), high=high.cpu().numpy(), mask=mask.cpu().numpy(), preds=preds.cpu().detach().numpy())
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

            if batch_idx % 50 == 0:
                lr_now = optimizer.param_groups[0]['lr']
                print(f"Epoch [{epoch}/{epochs}] Batch [{batch_idx}] loss={loss.item():.6f} lr={lr_now:.2e}")
                logger.log_batch(epoch, batch_idx, loss.item(), optimizer.param_groups[0]['lr'])

        avg_train_loss = running_loss / n_batches if n_batches > 0 else float('nan')

        # Validation
        model.eval()
        val_loss_acc = 0.0
        val_batches = 0
        val_psnr_acc = 0.0
        val_psnr_count = 0

        with torch.no_grad():
            for low_v, high_v, mask_v in val_loader:
                low_v = low_v.to(device); high_v = high_v.to(device); mask_v = mask_v.to(device)
                if mask_v.sum() < 1.0:
                    continue
                out_v = model(low_v)
                loss_v = criterion(out_v, high_v, mask_v)
                val_loss_acc += loss_v.item()
                val_batches += 1
                psnr_v = calculate_psnr_masked(out_v, high_v, mask_v)
                if not np.isnan(psnr_v):
                    val_psnr_acc += psnr_v
                    val_psnr_count += 1

        avg_val_loss = val_loss_acc / val_batches if val_batches > 0 else float('nan')
        avg_val_psnr = val_psnr_acc / val_psnr_count if val_psnr_count > 0 else float('nan')

        # scheduler step on validation loss
        scheduler.step(avg_val_loss if np.isfinite(avg_val_loss) else float('inf'))

        # logging
        lr_now = optimizer.param_groups[0]['lr']
        logger.log_epoch(epoch, avg_train_loss, avg_val_loss, None, avg_val_psnr, lr_now)

        elapsed = time.time() - epoch_start
        print(f"Epoch {epoch} summary: train_loss={avg_train_loss:.6f}, val_loss={avg_val_loss:.6f}, val_psnr={avg_val_psnr:.2f} dB, time={elapsed/60:.2f} min")

        # checkpoint by best val PSNR
        if np.isfinite(avg_val_psnr) and avg_val_psnr > best_val_psnr:
            best_val_psnr = avg_val_psnr
            best_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_psnr': best_val_psnr,
                'config': config
            }
            torch.save(best_checkpoint, save_path)
            print(f"Novo melhor PSNR={best_val_psnr:.2f} dB -> modelo salvo: {save_path}")

    print("Treinamento finalizado.")
    return model, logger.get_log_file(), best_checkpoint

# ---------------------------
# Inference / generation (tile-based)
# ---------------------------
def generate_same_res(model_ckpt_path, input_raster, output_raster, tile_size=256, overlap=0.5, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(model_ckpt_path, map_location=device)
    cfg = ckpt.get('config', {})
    base_filters = cfg.get('base_filters', 32)
    depth = cfg.get('depth', 4)
    norm_type = cfg.get('norm_type', 'group')
    model = SymmetricUNet(in_channels=1, out_channels=1, base_filters=base_filters, depth=depth, norm_type=norm_type).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    with rasterio.open(input_raster) as src:
        img = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata
        valid_mask = np.isfinite(img)
        if nodata is not None:
            valid_mask &= (img != nodata)
        if not np.any(valid_mask):
            print("Imagem sem dados válidos! Abortando.")
            return
        valid = img[valid_mask]
        p2, p98 = np.percentile(valid, [2,98])
        eps = 1e-6
        if (p98 - p2) <= eps:
            p98 = p2 + eps
        img_norm = np.zeros_like(img, dtype=np.float32)
        img_norm[valid_mask] = np.clip((img[valid_mask] - p2) / (p98 - p2), 0, 1)

        out_h = img.shape[0]; out_w = img.shape[1]
        output = np.zeros((out_h, out_w), dtype=np.float32)
        weight = np.zeros_like(output, dtype=np.float32)
        hann = np.outer(np.hanning(tile_size), np.hanning(tile_size))
        step = int(tile_size * (1-overlap))
        rows = list(range(0, img.shape[0], step))
        cols = list(range(0, img.shape[1], step))
        total = len(rows) * len(cols)
        processed = 0

        for i in rows:
            for j in cols:
                i_end = min(i+tile_size, img.shape[0]); j_end = min(j+tile_size, img.shape[1])
                tile = img_norm[i:i_end, j:j_end]; th, tw = tile.shape
                pad = np.zeros((tile_size, tile_size), dtype=np.float32); pad[:th,:tw] = tile
                weight_mask = np.zeros((tile_size, tile_size), dtype=np.float32); weight_mask[:th,:tw] = hann[:th,:tw]
                t_tensor = torch.from_numpy(pad).unsqueeze(0).unsqueeze(0).to(device).float()
                with torch.no_grad():
                    out_tile = model(t_tensor).cpu().numpy().squeeze()
                out_valid = out_tile[:th, :tw]
                w_valid = weight_mask[:th, :tw]
                output[i:i_end, j:j_end] += out_valid * w_valid
                weight[i:i_end, j:j_end] += w_valid
                processed += 1
                if processed % 50 == 0 or processed == total:
                    print(f"Tiles processados: {processed}/{total}")

        valid_w = weight > 1e-6
        output[valid_w] /= weight[valid_w]
        output = output * (p98 - p2 + 1e-8) + p2
        if nodata is not None:
            output[~valid_w] = float(nodata)
        profile.update(dtype='float32', count=1)
        os.makedirs(os.path.dirname(output_raster) or ".", exist_ok=True)
        with rasterio.open(output_raster, 'w', **profile) as dst:
            dst.write(output.astype(np.float32), 1)
        print(f"Output salvo: {output_raster}")

# ---------------------------
# Execução principal (ajuste caminhos)
# ---------------------------
if __name__ == "__main__":
    # ajuste caminhos para seus dados
    train_low_files = ["dados/anadem_5m.tif"]   # exemplo
    train_high_files = ["dados/geosampa_5m.tif"]
    val_low_files = ["dados/ANADEM_Recorte_IPT_5m.tif"]       # validação - região diferente
    val_high_files = ["dados/GEOSAMPA_Recorte_IPT.tif"]


    os.makedirs("model", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    model, log_file, best_ckpt = train_and_validate(
        train_low_files, train_high_files,
        val_low_files, val_high_files,
        epochs=30,
        batch_size=2,
        patch_size=128,
        base_filters=32,
        depth=4,                # se pouca VRAM, reduzir base_filters ou depth
        lr=1e-4,
        weight_decay=1e-6,
        save_path="model/unet_sym_masked_adam_bestpsnr.pth",
        samples_per_file=1000,
        use_ssim=False,         # começar False; depois testar True
        norm_type='group',
        dropout=0.03
    )

    print("="*60)
    print("GERANDO GRÁFICOS")
    print("="*60)
    TrainingVisualizer.plot_training_curves(log_file)

    print("="*60)
    print("INFERÊNCIA (gerar raster melhorado)")
    print("="*60)
    if best_ckpt is not None:
        ckpt_path = "model/unet_sym_masked_adam_bestpsnr.pth"
    else:
        ckpt_path = None
    if ckpt_path and os.path.exists(ckpt_path):
        generate_same_res(ckpt_path, "dados/ANADEM_Recorte_IPT_5m.tif", "output/ANADEM_Recorte_IPT_5m_improved_unet.tif", tile_size=256, overlap=0.5)
    else:
        print("Nenhum checkpoint encontrado para inferência. Rode treino primeiro ou ajuste caminho.")
