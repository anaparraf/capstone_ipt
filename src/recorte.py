import rasterio
import os
from rasterio.mask import mask
from shapely.geometry import box
from rasterio.features import geometry_mask
import shutil
from rasterio.windows import from_bounds, transform as window_transform
import numpy as np

def recortar_raster_por_coordenadas(input_raster, minx, miny, maxx, maxy, output_raster):
    """
    Recorta um raster usando um retângulo de coordenadas geográficas.
    """
    bbox = box(minx, miny, maxx, maxy)
    geometries = [bbox]

    if not os.path.exists(input_raster):
        print(f"Erro: Arquivo de entrada não encontrado: {input_raster}")
        return None

    try:
        with rasterio.open(input_raster) as src:
            out_image, out_transform = mask(src, geometries, crop=True)
            out_meta = src.meta.copy()

            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })

            with rasterio.open(output_raster, "w", **out_meta) as dest:
                dest.write(out_image)

            print(f" Recorte salvo em: {output_raster}")
            print(f" Dimensões: {out_image.shape[1]} x {out_image.shape[2]}")
            return output_raster

    except ValueError as e:
        print(f"Erro ao recortar o raster: {e}")
        return None


data_dir = "dados"
anadem_path = os.path.join(data_dir, "ANADEM_AricanduvaBufferUTM.tif")
geosampa_path = os.path.join(data_dir, "MDTGeosampa_AricanduvaBufferUTM.tif")

minx_val = 343933.2525743812
maxx_val = 347833.2525743812
miny_val = 7390528.299063631
maxy_val = 7396378.299063631

minx_val2 = 349483.2525743812
maxx_val2 = 353683.2525743812
miny_val2 = 7387888.299063631
maxy_val2 = 7390588.299063631

# recorta/subtrai região de sao paulo
# Caminhos dos arquivos
tif_sp = r"D:\casptone\MDTGeosampa\MDT_sampa-ZSTD.tif"
tif_regiao = os.path.join(data_dir, "ANADEM_Recorte_IPT_5m.tif")
saida = "saopaulo_menos_teste.tif"

# Abrir os rasters e processar apenas a janela de sobreposição (evita alocar o raster inteiro)
with rasterio.open(tif_sp) as sp, rasterio.open(tif_regiao) as regiao:
    # Bounds da região menor (minx, miny, maxx, maxy)
    minx, miny, maxx, maxy = regiao.bounds

    # Calcular interseção com os bounds do raster grande
    sp_left, sp_bottom, sp_right, sp_top = sp.bounds
    ov_minx = max(minx, sp_left)
    ov_miny = max(miny, sp_bottom)
    ov_maxx = min(maxx, sp_right)
    ov_maxy = min(maxy, sp_top)

    if ov_minx >= ov_maxx or ov_miny >= ov_maxy:
        print("Sem sobreposição entre os rasters. Nenhuma alteração necessária.")
    else:
        # Janela que cobre apenas a área de sobreposição no raster grande
        win = from_bounds(ov_minx, ov_miny, ov_maxx, ov_maxy, transform=sp.transform)
        win = win.round_offsets().round_shape()  # alinhar a janela a pixels inteiros
        win_transform = window_transform(win, sp.transform)

        # Ler apenas a janela do raster grande
        sp_window_data = sp.read(1, window=win)
        print(f"Lido window {win} com shape {sp_window_data.shape}")

        # Geometria da área de sobreposição (usar a interseção)
        geom_overlap = [box(ov_minx, ov_miny, ov_maxx, ov_maxy)]

        # Criar máscara apenas na shape da janela (geometry_mask(..., invert=True) retorna True dentro da geometria)
        mask_regiao = geometry_mask(
            geom_overlap,
            transform=win_transform,
            invert=True,
            out_shape=(sp_window_data.shape[0], sp_window_data.shape[1])
        )

        # Definir valor de preenchimento: nodata do raster grande ou 0 se não definido
        fill_value = sp.nodata if sp.nodata is not None else 0

        # Aplicar a máscara: zera ONDE a geometria existe (mask_regiao == True)
        sp_window_data[mask_regiao] = fill_value

        # Copiar arquivo original para saída (evita leitura inteira) e escrever somente a janela modificada
        shutil.copyfile(tif_sp, saida)
        with rasterio.open(saida, 'r+') as dst:
            dst.write(sp_window_data, 1, window=win)

        print("✅ Recorte concluído (janela aplicada):", saida)