import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
import torch
import torch.nn.functional as F
import os
import time
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importar as classes do arquivo principal
from unet3 import ResolutionAwareUNet, generate_super_resolution

def evaluate_super_resolution(reference_path, predicted_path, nodata_value=None):
    """
    Avalia a qualidade da super resolução usando múltiplas métricas
    
    Args:
        reference_path: Caminho para imagem de referência (ground truth)
        predicted_path: Caminho para imagem predita
        nodata_value: Valor de NoData a ser ignorado
    
    Returns:
        dict: Dicionário com métricas calculadas
    """
    # Carregar imagens
    with rasterio.open(reference_path) as src:
        ref_img = src.read(1).astype(np.float32)
        if nodata_value is None:
            nodata_value = src.nodata

    with rasterio.open(predicted_path) as src:
        pred_img = src.read(1).astype(np.float32)

    # Verificar se as dimensões são compatíveis
    if ref_img.shape != pred_img.shape:
        print(f"Aviso: Dimensões diferentes - Ref: {ref_img.shape}, Pred: {pred_img.shape}")
        min_h = min(ref_img.shape[0], pred_img.shape[0])
        min_w = min(ref_img.shape[1], pred_img.shape[1])
        ref_img = ref_img[:min_h, :min_w]
        pred_img = pred_img[:min_h, :min_w]

    # Máscara de pixels válidos (NoData, NaN, inf, valores extremos)
    valid_mask = np.ones_like(ref_img, dtype=bool)
    if nodata_value is not None:
        valid_mask &= (ref_img != nodata_value) & (pred_img != nodata_value)
    valid_mask &= np.isfinite(ref_img) & np.isfinite(pred_img)
    # Remover valores extremos (outliers)
    ref_valid = ref_img[valid_mask]
    pred_valid = pred_img[valid_mask]
    if len(ref_valid) > 0:
        ref_q1, ref_q99 = np.percentile(ref_valid, [1, 99])
        pred_q1, pred_q99 = np.percentile(pred_valid, [1, 99])
        extreme_mask = (ref_valid >= ref_q1) & (ref_valid <= ref_q99) & (pred_valid >= pred_q1) & (pred_valid <= pred_q99)
        ref_valid = ref_valid[extreme_mask]
        pred_valid = pred_valid[extreme_mask]

    if not np.any(ref_valid) or not np.any(pred_valid):
        print("Erro: Nenhum pixel válido encontrado!")
        return None

    # Normalização robusta para [0, 1] (percentis)
    ref_min, ref_max = np.percentile(ref_valid, [2, 98])
    pred_min, pred_max = np.percentile(pred_valid, [2, 98])
    ref_norm = np.clip((ref_valid - ref_min) / (ref_max - ref_min + 1e-8), 0, 1) if ref_max > ref_min else np.zeros_like(ref_valid)
    pred_norm = np.clip((pred_valid - pred_min) / (pred_max - pred_min + 1e-8), 0, 1) if pred_max > pred_min else np.zeros_like(pred_valid)

    metrics = {}
    try:
        # Mean Squared Error
        metrics['MSE'] = mean_squared_error(ref_valid, pred_valid)
        metrics['MSE_norm'] = mean_squared_error(ref_norm, pred_norm)
        metrics['RMSE'] = np.sqrt(metrics['MSE'])
        metrics['MAE'] = np.mean(np.abs(ref_valid - pred_valid))
        metrics['MAE_norm'] = np.mean(np.abs(ref_norm - pred_norm))

        # Peak Signal-to-Noise Ratio
        metrics['PSNR'] = peak_signal_noise_ratio(ref_norm, pred_norm, data_range=1.0) if ref_max > ref_min else float('inf')

        # Structural Similarity Index
        # Para SSIM, reconstruir imagens 2D (opcional: pode ser só para pixels válidos)
        # Aqui, SSIM só se houver pixels suficientes
        if len(ref_valid) >= 49:
            metrics['SSIM'] = structural_similarity(ref_norm, pred_norm, data_range=1.0)
        else:
            metrics['SSIM'] = 0.0

        # Correlation coefficient
        if len(ref_valid) > 1:
            correlation_matrix = np.corrcoef(ref_valid, pred_valid)
            metrics['Correlation'] = correlation_matrix[0, 1] if not np.isnan(correlation_matrix[0, 1]) else 0.0
        else:
            metrics['Correlation'] = 0.0

        # Relative error
        metrics['Relative_Error'] = np.mean(np.abs(ref_valid - pred_valid) / (np.abs(ref_valid) + 1e-8))

        # R2 Score
        ss_res = np.sum((ref_valid - pred_valid) ** 2)
        ss_tot = np.sum((ref_valid - np.mean(ref_valid)) ** 2)
        metrics['R2'] = 1 - ss_res / (ss_tot + 1e-8)

        # Bias
        metrics['Bias'] = np.mean(pred_valid - ref_valid)

        # Mediana dos erros
        metrics['Median_AE'] = np.median(np.abs(ref_valid - pred_valid))

    except Exception as e:
        print(f"Erro ao calcular métricas: {e}")
        return None

    return metrics


def visualize_comparison(reference_path, predicted_path, low_res_path=None, 
                        save_path=None, figsize=(15, 5)):
    """
    Visualiza comparação entre imagens de referência, predita e opcionalmente baixa resolução
    """
    fig, axes = plt.subplots(1, 3 if low_res_path else 2, figsize=figsize)
    
    # Carregar e mostrar imagem de referência
    with rasterio.open(reference_path) as src:
        ref_img = src.read(1)
        show(src, ax=axes[0], title='Referência (Ground Truth)')
    
    # Carregar e mostrar imagem predita
    with rasterio.open(predicted_path) as src:
        pred_img = src.read(1)
        show(src, ax=axes[1], title='Super Resolução (Predita)')
    
    # Se disponível, mostrar imagem de baixa resolução
    if low_res_path:
        with rasterio.open(low_res_path) as src:
            show(src, ax=axes[2], title='Baixa Resolução (Input)')
    
    # Calcular e mostrar métricas
    metrics = evaluate_super_resolution(reference_path, predicted_path)
    if metrics:
        metrics_text = f"MSE: {metrics['MSE']:.4f}\n"
        metrics_text += f"PSNR: {metrics['PSNR']:.2f} dB\n"
        metrics_text += f"SSIM: {metrics['SSIM']:.4f}\n"
        metrics_text += f"Correlation: {metrics['Correlation']:.4f}"
        
        plt.figtext(0.02, 0.02, metrics_text, fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualização salva em: {save_path}")
    
    plt.show()
    
    return metrics


def plot_training_curves(loss_history, save_path=None):
    """
    Plota curvas de treinamento
    """
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(np.log(loss_history))
    plt.title('Training Loss (Log Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('Log(Loss)')
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def batch_evaluate_resolutions(model_path, test_files, target_resolutions, output_dir="evaluation_results"):
    """
    Avalia modelo para múltiplas resoluções
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    
    for resolution in target_resolutions:
        print(f"\n🔄 Avaliando resolução: {resolution}m")
        resolution_results = []
        
        for i, test_file in enumerate(test_files):
            output_path = os.path.join(output_dir, f"test_{i}_res_{resolution}m.tif")
            
            # Gerar super resolução
            try:
                generate_super_resolution(model_path, test_file, output_path, 
                                        target_resolution=resolution)
                
                # Se houver arquivo de referência, avaliar
                reference_file = test_file.replace("low_res", "high_res")  # Assumindo convenção de nomes
                if os.path.exists(reference_file):
                    metrics = evaluate_super_resolution(reference_file, output_path)
                    if metrics:
                        metrics['file'] = test_file
                        metrics['resolution'] = resolution
                        resolution_results.append(metrics)
                        print(f"  ✅ Arquivo {i+1}: PSNR={metrics['PSNR']:.2f}dB, SSIM={metrics['SSIM']:.4f}")
                
            except Exception as e:
                print(f"  ❌ Erro no arquivo {i+1}: {e}")
        
        results[resolution] = resolution_results
    
    # Resumo dos resultados
    print("\n📊 RESUMO DOS RESULTADOS:")
    print("-" * 60)
    for resolution, res_list in results.items():
        if res_list:
            avg_psnr = np.mean([r['PSNR'] for r in res_list if np.isfinite(r['PSNR'])])
            avg_ssim = np.mean([r['SSIM'] for r in res_list])
            avg_mse = np.mean([r['MSE'] for r in res_list])
            print(f"Resolução {resolution}m: PSNR={avg_psnr:.2f}dB, SSIM={avg_ssim:.4f}, MSE={avg_mse:.6f}")
    
    return results


def create_resolution_comparison_plot(results_dict, save_path=None):
    """
    Cria gráfico comparativo de métricas por resolução
    """
    resolutions = []
    psnr_values = []
    ssim_values = []
    mse_values = []
    
    for resolution, res_list in results_dict.items():
        if res_list:
            resolutions.append(resolution)
            psnr_values.append(np.mean([r['PSNR'] for r in res_list if np.isfinite(r['PSNR'])]))
            ssim_values.append(np.mean([r['SSIM'] for r in res_list]))
            mse_values.append(np.mean([r['MSE'] for r in res_list]))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # PSNR
    axes[0].plot(resolutions, psnr_values, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Resolução Alvo (m)')
    axes[0].set_ylabel('PSNR (dB)')
    axes[0].set_title('PSNR vs Resolução')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale('log')
    
    # SSIM
    axes[1].plot(resolutions, ssim_values, 'ro-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Resolução Alvo (m)')
    axes[1].set_ylabel('SSIM')
    axes[1].set_title('SSIM vs Resolução')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale('log')
    
    # MSE
    axes[2].plot(resolutions, mse_values, 'go-', linewidth=2, markersize=8)
    axes[2].set_xlabel('Resolução Alvo (m)')
    axes[2].set_ylabel('MSE')
    axes[2].set_title('MSE vs Resolução')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xscale('log')
    axes[2].set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico comparativo salvo em: {save_path}")
    
    plt.show()


def analyze_raster_properties(raster_path):
    """
    Analisa propriedades de um raster
    """
    with rasterio.open(raster_path) as src:
        print(f"📁 Arquivo: {os.path.basename(raster_path)}")
        print(f"   Dimensões: {src.width} x {src.height}")
        print(f"   Bandas: {src.count}")
        print(f"   Tipo de dados: {src.dtypes[0]}")
        print(f"   CRS: {src.crs}")
        print(f"   NoData: {src.nodata}")
        
        # Resolução
        res_x = abs(src.transform.a)
        res_y = abs(src.transform.e)
        print(f"   Resolução: {res_x:.4f} x {res_y:.4f} metros/pixel")
        
        # Extensão
        bounds = src.bounds
        print(f"   Extensão: {bounds}")
        
        # Estatísticas dos dados
        data = src.read(1)
        valid_data = data[data != src.nodata] if src.nodata is not None else data
        
        if len(valid_data) > 0:
            print(f"   Min: {valid_data.min():.4f}")
            print(f"   Max: {valid_data.max():.4f}")
            print(f"   Média: {valid_data.mean():.4f}")
            print(f"   Std: {valid_data.std():.4f}")
            print(f"   Pixels válidos: {len(valid_data)}/{data.size} ({100*len(valid_data)/data.size:.1f}%)")
        
        print("-" * 50)


class SuperResolutionBenchmark:
    """
    Classe para benchmark completo de super resolução
    """
    def __init__(self, model_path):
        self.model_path = model_path
        self.results = {}
    
    def run_benchmark(self, test_files, target_resolutions, reference_files=None):
        """
        Executa benchmark completo
        """
        print("🚀 Iniciando benchmark de super resolução...")
        print(f"📦 Modelo: {self.model_path}")
        print(f"🎯 Resoluções alvo: {target_resolutions}")
        print(f"📋 Arquivos de teste: {len(test_files)}")
        
        # Analisar arquivos de entrada
        print("\n📊 ANÁLISE DOS ARQUIVOS DE ENTRADA:")
        for i, file_path in enumerate(test_files):
            print(f"\nArquivo {i+1}:")
            analyze_raster_properties(file_path)
        
        # Executar avaliação para todas as resoluções
        self.results = batch_evaluate_resolutions(
            self.model_path, test_files, target_resolutions
        )
        
        # Criar visualizações
        print("\n📈 Gerando visualizações...")
        create_resolution_comparison_plot(self.results, "evaluation_results/resolution_comparison.png")
        
        return self.results
    
    def generate_report(self, output_file="benchmark_report.txt"):
        """
        Gera relatório detalhado
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("RELATÓRIO DE BENCHMARK - SUPER RESOLUÇÃO\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Modelo avaliado: {self.model_path}\n")
            f.write(f"Data: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for resolution, results_list in self.results.items():
                f.write(f"\nRESOLUÇÃO ALVO: {resolution}m\n")
                f.write("-" * 30 + "\n")
                
                if results_list:
                    metrics_summary = {
                        'PSNR': [r['PSNR'] for r in results_list if np.isfinite(r['PSNR'])],
                        'SSIM': [r['SSIM'] for r in results_list],
                        'MSE': [r['MSE'] for r in results_list],
                        'Correlation': [r['Correlation'] for r in results_list]
                    }
                    
                    for metric, values in metrics_summary.items():
                        if values:
                            f.write(f"{metric}:\n")
                            f.write(f"  Média: {np.mean(values):.6f}\n")
                            f.write(f"  Desvio: {np.std(values):.6f}\n")
                            f.write(f"  Min: {np.min(values):.6f}\n")
                            f.write(f"  Max: {np.max(values):.6f}\n")
                    
                    f.write(f"\nNúmero de arquivos processados: {len(results_list)}\n")
                else:
                    f.write("Nenhum resultado válido para esta resolução.\n")
        
        print(f"📄 Relatório salvo em: {output_file}")


# Exemplo de uso das funções de avaliação
def example_evaluation():
    """
    Exemplo de como usar as funções de avaliação
    """
    # Caminhos dos arquivos (substitua pelos seus)
    model_path = "model/adaptive_unet_5m_v2.pth"
    test_files = ["dados/ANADEM_Recorte_IPT.tif"]  # Arquivos de baixa resolução para teste
    # target_resolutions = [0.5, 1.0, 2.0, 5.0]  # Resoluções alvo em metros
    target_resolutions = [0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6,  7, 8, 9, 10, 11, 12, 13, 14, 15] 
    # Executar benchmark
    benchmark = SuperResolutionBenchmark(model_path)
    results = benchmark.run_benchmark(test_files, target_resolutions)
    
    # Gerar relatório
    benchmark.generate_report("evaluation_results/meu_benchmark_report.txt")
    
    # Visualização específica (se você tiver arquivos de referência)
    # visualize_comparison(
    #     "dados/GEOSAMPA_Recorte_IPT.tif",
    #     "evaluation_results/test_0_res_0.5m.tif",
    #     "dados/ANADEM_Recorte_IPT.tif",
    #     save_path="comparison_plot.png"
    # )

if __name__ == "__main__":
    example_evaluation()