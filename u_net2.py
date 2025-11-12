import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

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

# -------------------------
# Configuração inicial
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# ================================
# Dataset com normalização Z-score
# ================================
class DEMRefinementDataset(Dataset):
    """
    Dataset para refinamento de DEM: duas entradas de mesma resolução (5m)
    - input1: ANADEM (menos detalhado, 5m)
    - input2: GeoSampa (mais detalhado, 5m)
    - target: GeoSampa (ground truth para treinamento)
    
    MODIFICAÇÃO: Normalização Z-score para preservar feições sutis
    """
    def __init__(self, input1_files, input2_files, target_files=None, transform=None, patch_size=128):
        self.input1_datasets = [rasterio.open(f) for f in input1_files]
        self.input2_datasets = [rasterio.open(f) for f in input2_files]
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
        return 20000  # Aumentado para mais variabilidade

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

        # *** NOVA NORMALIZAÇÃO Z-SCORE ***
        img1_norm = self._normalize_zscore(img1, src1.nodata)
        img2_norm = self._normalize_zscore(img2, src2.nodata)
        img_target_norm = self._normalize_zscore(img_target, tgt.nodata)

        # Augmentation que preserva estruturas
        img1_norm, img2_norm, img_target_norm = self._augment_preserve_structure(
            img1_norm, img2_norm, img_target_norm
        )

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

    def _normalize_zscore(self, img, nodata_value):
        """
        Normalização Z-score: preserva variações relativas (essencial para rios!)
        
        Vantagem sobre percentis:
        - Rios com variação de 2-5m mantêm contraste relativo
        - Não esmaga feições sutis
        - Média=0, desvio=1 → melhor para redes neurais
        """
        if nodata_value is not None:
            valid_mask = img != nodata_value
        else:
            valid_mask = np.isfinite(img)

        if not np.any(valid_mask):
            return np.zeros_like(img, dtype=np.float32)

        valid_data = img[valid_mask]
        mean = np.mean(valid_data)
        std = np.std(valid_data)

        img_norm = np.zeros_like(img, dtype=np.float32)
        
        if std > 1e-6:  # Evita divisão por zero
            # Normalização Z-score
            img_norm[valid_mask] = (img[valid_mask] - mean) / std
            
            # Clip extremos (mas preserva 99.7% dos dados = ±3σ)
            img_norm = np.clip(img_norm, -3, 3)
            
            # Rescale para [0, 1] para compatibilidade com a rede
            img_norm = (img_norm + 3) / 6
        else:
            # Região plana (sem variação)
            img_norm[valid_mask] = 0.5

        return img_norm

    def _augment_preserve_structure(self, img1, img2, target):
        """
        Data augmentation que PRESERVA estruturas topográficas
        - SEM ruído Gaussiano (destrói feições finas)
        - Apenas transformações geométricas
        """
        # Flip horizontal (50% chance)
        if np.random.random() > 0.5:
            img1 = np.flip(img1, axis=1).copy()
            img2 = np.flip(img2, axis=1).copy()
            target = np.flip(target, axis=1).copy()
        
        # Flip vertical (50% chance)
        if np.random.random() > 0.5:
            img1 = np.flip(img1, axis=0).copy()
            img2 = np.flip(img2, axis=0).copy()
            target = np.flip(target, axis=0).copy()
        
        # Rotação 90° (0°, 90°, 180°, 270°)
        k = np.random.randint(0, 4)
        if k > 0:
            img1 = np.rot90(img1, k).copy()
            img2 = np.rot90(img2, k).copy()
            target = np.rot90(target, k).copy()
        
        return img1, img2, target

    def __del__(self):
        for dataset in (self.input1_datasets + self.input2_datasets + self.target_datasets):
            if hasattr(dataset, 'close'):
                dataset.close()


# ================================
# Model / Blocks (mantidos)
# ================================
class ConvBlock(nn.Module):
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
    def __init__(self, in_channels=2, out_channels=1, base_filters=32, depth=4,
                 kernel_size=3, pool_size=2, dropout_rate=0.1, inference_mode=False):
        super(DEMRefinementUNet, self).__init__()

        self.depth = depth
        self.inference_mode = inference_mode

        actual_in_channels = in_channels

        # Encoder
        self.down_blocks = nn.ModuleList()
        for i in range(depth):
            in_ch = actual_in_channels if i == 0 else base_filters * (2 ** (i - 1))
            out_ch = base_filters * (2 ** i)
            use_dropout = True
            self.down_blocks.append(ConvBlock(in_ch, out_ch, kernel_size, use_dropout, dropout_rate))

        self.pool = nn.MaxPool2d(pool_size)

        # Bottleneck
        bottleneck_channels = base_filters * (2 ** (depth - 1))
        self.bottleneck = ConvBlock(bottleneck_channels, bottleneck_channels * 2,
                                    kernel_size, use_dropout=True, dropout_rate=dropout_rate)

        # Decoder
        self.up_transpose_blocks = nn.ModuleList()
        self.up_conv_blocks = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            in_ch = bottleneck_channels * 2 if i == depth - 1 else base_filters * (2 ** (i + 1))
            out_ch = base_filters * (2 ** i)
            self.up_transpose_blocks.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2))
            self.up_conv_blocks.append(ConvBlock(out_ch * 2, out_ch, kernel_size,
                                                use_dropout=False, dropout_rate=dropout_rate))

        self.final = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down_block in self.down_blocks:
            x = down_block(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections = list(reversed(skip_connections))
        for up_transpose, up_conv, skip_connection in zip(self.up_transpose_blocks,
                                                           self.up_conv_blocks,
                                                           skip_connections):
            x = up_transpose(x)
            if x.shape[2:] != skip_connection.shape[2:]:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat((x, skip_connection), dim=1)
            x = up_conv(x)

        x = self.final(x)
        return x


# ================================
# Loss com Curvature (detecta rios!)
# ================================
class TopographicLoss(nn.Module):
    """
    Loss especializada para topografia:
    - MSE: erro absoluto
    - L1: erro robusto
    - Gradient: bordas/declives
    - Curvature (NOVO): detecta concavidades (rios/vales)
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        l1_loss = self.l1(pred, target)
        grad_loss = self._gradient_loss(pred, target)
        curv_loss = self._curvature_loss(pred, target)
        
        # Pesos ajustados: curvature é crítico para rios!
        return 0.3 * mse_loss + 0.2 * l1_loss + 0.3 * grad_loss + 0.2 * curv_loss

    def _gradient_loss(self, pred, target):
        """Detecta bordas e declives"""
        def gradient(img):
            grad_x = img[:, :, :, 1:] - img[:, :, :, :-1]
            grad_y = img[:, :, 1:, :] - img[:, :, :-1, :]
            return grad_x, grad_y

        pred_grad_x, pred_grad_y = gradient(pred)
        target_grad_x, target_grad_y = gradient(target)

        loss_x = F.l1_loss(pred_grad_x, target_grad_x)
        loss_y = F.l1_loss(pred_grad_y, target_grad_y)

        return loss_x + loss_y

    def _curvature_loss(self, pred, target):
        """
        Laplaciano (2ª derivada): detecta concavidades
        
        Rios = regiões côncavas (Laplaciano positivo)
        Cumeadas = regiões convexas (Laplaciano negativo)
        """
        def laplacian(img):
            # Kernel Laplaciano discreto
            kernel = torch.tensor([[0, 1, 0],
                                   [1, -4, 1],
                                   [0, 1, 0]], 
                                  dtype=img.dtype, device=img.device)
            kernel = kernel.view(1, 1, 3, 3)
            
            # Replicar kernel para todos os canais
            kernel = kernel.repeat(img.size(1), 1, 1, 1)
            
            # Convolução (groups=channels para aplicar por canal)
            lapl = F.conv2d(img, kernel, padding=1, groups=img.size(1))
            return lapl
        
        pred_lapl = laplacian(pred)
        target_lapl = laplacian(target)
        
        return F.l1_loss(pred_lapl, target_lapl)


# ================================
# Função para visualizar patches durante treino
# ================================
def visualize_batch(inputs, targets, predictions, epoch, batch_idx, save_dir="debug_patches"):
    """Salva visualizações para debug"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Primeiro exemplo do batch
    anadem = inputs[0, 0].cpu().numpy()
    geosampa_in = inputs[0, 1].cpu().numpy()
    target = targets[0, 0].cpu().numpy()
    pred = predictions[0, 0].detach().cpu().numpy()
    
    axes[0].imshow(anadem, cmap='terrain')
    axes[0].set_title('Input ANADEM')
    axes[0].axis('off')
    
    axes[1].imshow(geosampa_in, cmap='terrain')
    axes[1].set_title('Input GeoSampa')
    axes[1].axis('off')
    
    axes[2].imshow(target, cmap='terrain')
    axes[2].set_title('Target')
    axes[2].axis('off')
    
    axes[3].imshow(pred, cmap='terrain')
    axes[3].set_title('Prediction')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/epoch{epoch:03d}_batch{batch_idx:04d}.png", dpi=100)
    plt.close()


# ================================
# Treinamento com coleta de métricas
# ================================
def train_dem_refinement(input1_files, input2_files, target_files=None,
                         epochs=100, batch_size=4, learning_rate=1e-4,
                         patch_size=128, save_path="model/dem_refinement_unet.pth",
                         output_metrics_dir="output_metrics"):
    print("Iniciando treinamento de refinamento de DEM")
    print(f"MODIFICAÇÕES: Z-score normalization + Curvature loss + patch_size={patch_size}")
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
    criterion = TopographicLoss().to(device)  # *** NOVA LOSS ***

    model.train()
    best_loss = float('inf')

    # Histórico de métricas
    history = {
        "epoch": [],
        "total_loss": [],
        "mse_loss": [],
        "l1_loss": [],
        "grad_loss": [],
        "ssim_loss": [],
        "psnr": [],
        "lr": []
    }

    for epoch in range(epochs):
        epoch_total_loss = 0.0
        epoch_mse = 0.0
        epoch_l1 = 0.0
        epoch_grad = 0.0
        epoch_ssim = 0.0
        epoch_psnr = 0.0
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

            # Cálculos de componentes para logging
            mse_val = F.mse_loss(predictions, targets).item()
            l1_val = F.l1_loss(predictions, targets).item()
            
            # Gradient loss
            def compute_gradient_loss(pred, target):
                def gradient(img):
                    grad_x = img[:, :, :, 1:] - img[:, :, :, :-1]
                    grad_y = img[:, :, 1:, :] - img[:, :, :-1, :]
                    return grad_x, grad_y
                pred_grad_x, pred_grad_y = gradient(pred)
                target_grad_x, target_grad_y = gradient(target)
                loss_x = F.l1_loss(pred_grad_x, target_grad_x)
                loss_y = F.l1_loss(pred_grad_y, target_grad_y)
                return loss_x + loss_y
            
            grad_val = compute_gradient_loss(predictions, targets).item()

            # PSNR / SSIM
            preds_np = predictions.detach().cpu().numpy()
            targs_np = targets.detach().cpu().numpy()
            batch_ssim = 0.0
            batch_psnr = 0.0
            B = preds_np.shape[0]
            for b in range(B):
                p = np.squeeze(preds_np[b])
                t = np.squeeze(targs_np[b])
                try:
                    batch_psnr += sk_psnr(t, p, data_range=1.0)
                except Exception:
                    batch_psnr += 0.0
                try:
                    batch_ssim += 1.0 - sk_ssim(t, p, data_range=1.0)
                except Exception:
                    batch_ssim += 1.0

            batch_ssim /= B
            batch_psnr /= B

            epoch_total_loss += loss.item()
            epoch_mse += mse_val
            epoch_l1 += l1_val
            epoch_grad += grad_val
            epoch_ssim += batch_ssim
            epoch_psnr += batch_psnr

            num_batches += 1

            # Visualizar patches a cada 100 batches
            if batch_idx % 100 == 0:
                print(f"Época [{epoch+1}/{epochs}], Batch [{batch_idx}], Loss: {loss.item():.6f}")
                # visualize_batch(inputs, targets, predictions, epoch, batch_idx)

        # Médias por época
        avg_total = epoch_total_loss / num_batches if num_batches > 0 else float('inf')
        avg_mse = epoch_mse / num_batches
        avg_l1 = epoch_l1 / num_batches
        avg_grad = epoch_grad / num_batches
        avg_ssim = epoch_ssim / num_batches
        avg_psnr = epoch_psnr / num_batches
        lr = optimizer.param_groups[0]['lr']

        # Atualiza scheduler e salva modelo se melhor
        scheduler.step(avg_total)

        if avg_total < best_loss:
            best_loss = avg_total
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, save_path)
            print(f"✅ Modelo salvo com loss: {best_loss:.6f}")

        history["epoch"].append(epoch + 1)
        history["total_loss"].append(avg_total)
        history["mse_loss"].append(avg_mse)
        history["l1_loss"].append(avg_l1)
        history["grad_loss"].append(avg_grad)
        history["ssim_loss"].append(avg_ssim)
        history["psnr"].append(avg_psnr)
        history["lr"].append(lr)

        print(f"Época [{epoch+1}/{epochs}] - Loss: {avg_total:.6f} | PSNR: {avg_psnr:.3f} dB | SSIM_loss: {avg_ssim:.4f} | LR: {lr:.6f}")

    elapsed = time.time() - start_time
    print(f"Tempo total de treinamento: {elapsed/60:.2f} minutos")

    # Salvar métricas e plots
    save_metrics_and_plots(history, output_dir=output_metrics_dir)

    return model, history


# ================================
# Funções de salvamento e plot (mantidas)
# ================================
def save_plot(x, y, title, ylabel, color, filename, output_dir="output_metrics", fill=False, marker=None):
    plt.figure(figsize=(7,4))
    plt.plot(x, y, color=color, label=title, marker=marker)
    if fill:
        plt.fill_between(x, y, alpha=0.2, color=color)
    plt.xlabel("Época")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, filename)
    plt.savefig(pdf_path, format='pdf')
    plt.close()
    print(f"✅ Gráfico salvo: {pdf_path}")


def save_metrics_and_plots(history, output_dir="output_metrics"):
    os.makedirs(output_dir, exist_ok=True)
    # CSV
    df = pd.DataFrame(history)
    csv_path = os.path.join(output_dir, "metricas_treinamento.csv")
    df.to_csv(csv_path, index=False)
    print(f"📄 Métricas salvas em: {csv_path}")

    # Plots individuais
    save_plot(history["epoch"], history["total_loss"], "Loss Total por Época", "Loss", "blue", "01_total_loss.pdf", output_dir, fill=True)
    save_plot(history["epoch"], history["mse_loss"], "Mean Squared Error", "MSE", "green", "02_mse_loss.pdf", output_dir)
    save_plot(history["epoch"], history["l1_loss"], "L1 (MAE)", "L1 Loss", "orange", "03_l1_loss.pdf", output_dir)
    save_plot(history["epoch"], history["grad_loss"], "Gradient Loss", "Grad Loss", "red", "04_grad_loss.pdf", output_dir)
    save_plot(history["epoch"], history["ssim_loss"], "SSIM Loss", "SSIM Loss", "purple", "05_ssim_loss.pdf", output_dir)
    save_plot(history["epoch"], history["lr"], "Learning Rate", "LR", "brown", "06_learning_rate.pdf", output_dir)
    save_plot(history["epoch"], history["psnr"], "PSNR (dB)", "PSNR (dB)", "darkgreen", "07_psnr.pdf", output_dir, marker='o')

    # Gráfico consolidado
    fig, axes = plt.subplots(4, 2, figsize=(12, 10))
    fig.suptitle("Treinamento U-Net (Z-score + Curvature Loss)", fontsize=14, fontweight='bold')

    axes[0, 0].plot(history["epoch"], history["total_loss"], color='blue', label='Total Loss')
    axes[0, 0].fill_between(history["epoch"], history["total_loss"], alpha=0.2, color='blue')
    axes[0, 0].set_title("Loss Total"); axes[0, 0].legend()

    axes[0, 1].plot(history["epoch"], history["mse_loss"], color='green', label='MSE Loss')
    axes[0, 1].set_title("Mean Squared Error"); axes[0, 1].legend()

    axes[1, 0].plot(history["epoch"], history["l1_loss"], color='orange', label='L1 Loss')
    axes[1, 0].set_title("L1 (MAE)"); axes[1, 0].legend()

    axes[1, 1].plot(history["epoch"], history["grad_loss"], color='red', label='Gradient Loss')
    axes[1, 1].set_title("Gradient Loss"); axes[1, 1].legend()

    axes[2, 0].plot(history["epoch"], history["ssim_loss"], color='purple', label='SSIM Loss')
    axes[2, 0].set_title("SSIM Loss"); axes[2, 0].legend()

    axes[2, 1].plot(history["epoch"], history["lr"], color='brown', label='Learning Rate')
    axes[2, 1].set_title("Learning Rate"); axes[2, 1].legend()

    axes[3, 0].plot(history["epoch"], history["psnr"], color='darkgreen', marker='o', label='PSNR')
    axes[3, 0].set_title("PSNR (dB)"); axes[3, 0].legend()

    axes[3, 1].axis('off')

    for ax in axes.flat:
        ax.set_xlabel("Época")
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    consolidado_path = os.path.join(output_dir, "00_metricas_consolidadas.pdf")
    plt.savefig(consolidado_path, format='pdf')
    plt.close()
    print(f"📊 Gráfico consolidado salvo: {consolidado_path}")


# ================================
# Geração de DEM refinado
# ================================
def generate_refined_dem(
    model_path, input_anadem_path, output_path,
    device=None, tile_size=256, overlap=0.5
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device)

    model = DEMRefinementUNet(in_channels=2, out_channels=1, base_filters=32, depth=4).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("Modelo carregado para refinamento de DEM (modo inferência)")

    with rasterio.open(input_anadem_path) as src:
        img = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata

        print(f"Dimensões ANADEM: {img.shape}")

        # *** NORMALIZAÇÃO Z-SCORE (mesma do treinamento) ***
        valid_mask = np.isfinite(img)
        if nodata is not None:
            valid_mask &= (img != nodata)

        if not np.any(valid_mask):
            print("ERRO: Imagem sem dados válidos!")
            return

        valid_data = img[valid_mask]
        mean = np.mean(valid_data)
        std = np.std(valid_data)

        print(f"Stats ANADEM: mean={mean:.2f}, std={std:.2f}")

        img_norm = np.zeros_like(img)
        if std > 1e-6:
            img_norm[valid_mask] = (img[valid_mask] - mean) / std
            img_norm = np.clip(img_norm, -3, 3)
            img_norm = (img_norm + 3) / 6
        else:
            img_norm[valid_mask] = 0.5

        out_height, out_width = img.shape
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

                tile = img_norm[i:i_end, j:j_end]
                tile_h, tile_w = tile.shape

                # Padding se necessário
                tile_padded = np.zeros((tile_size, tile_size), dtype=np.float32)
                tile_padded[:tile_h, :tile_w] = tile

                weight_mask = np.zeros((tile_size, tile_size), dtype=np.float32)
                weight_mask[:tile_h, :tile_w] = hann_window[:tile_h, :tile_w]

                # Duplicar o canal ANADEM (modelo foi treinado com 2 canais)
                tile_input = torch.stack([
                    torch.from_numpy(tile_padded),
                    torch.from_numpy(tile_padded)
                ]).unsqueeze(0).to(device).float()  # Shape: [1, 2, H, W]

                with torch.no_grad():
                    tile_out = model(tile_input)
                    tile_out_np = tile_out.cpu().numpy().squeeze()

                # Extrair parte válida
                tile_out_valid = tile_out_np[:tile_h, :tile_w]
                weights_valid = weight_mask[:tile_h, :tile_w]

                # Acumular
                output_array[i:i_end, j:j_end] += tile_out_valid * weights_valid
                weight_array[i:i_end, j:j_end] += weights_valid

                processed += 1
                if processed % 10 == 0:
                    print(f"Progresso: {processed}/{total_tiles} tiles")

        # Normalizar por pesos
        valid_weights = weight_array > 1e-6
        if np.any(valid_weights):
            output_array[valid_weights] /= weight_array[valid_weights]

        # *** DESNORMALIZAÇÃO Z-SCORE ***
        # Reverter: (x * 6) - 3 = z-score, depois z * std + mean = valor original
        output_array = (output_array * 6) - 3  # Volta para z-score
        output_array = output_array * std + mean  # Volta para escala original

        # Marcar nodata
        if nodata is not None:
            output_array[~valid_weights] = float(nodata)

        # Salvar
        profile.update({'dtype': 'float32'})
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(output_array.astype('float32'), 1)

        print(f"✅ DEM refinado gerado: {output_path}")


# ================================
# Execução 
# ================================
if __name__ == "__main__":
    # Caminhos dos arquivos (adapte conforme necessário)
    input1_files = ["dados/geosampa_30m_reamostrado_5m.tif"]  # ANADEM menos detalhado
    input2_files = ["dados/geosampa_5m_reprojetado.tif"]  # GeoSampa mais detalhado

    # ========================================

    model, history = train_dem_refinement(
        input1_files, input2_files,
        epochs=120,
        batch_size=4,
        patch_size=128,  # Reduzido de 256
        save_path="model/dem_refinement_f.pth",
        output_metrics_dir="output_metrics_zscore"
    )

    # ========================================
    # INFERÊNCIA
    # ========================================
    print("\n" + "="*60)
    print("INICIANDO INFERÊNCIA")
    print("="*60 + "\n")
    
    generate_refined_dem(
        "model/dem_refinement_f.pth",
        input_anadem_path="dados/ANADEM_Recorte_IPT_5m.tif",
        output_path="output/anadem_refined_f.tif",
        tile_size=256,
        overlap=0.5
    )
    
    print("\n" + "="*60)
    print("TREINAMENTO E INFERÊNCIA CONCLUÍDOS!")