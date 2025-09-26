import rasterio
import os
from rasterio.mask import mask
from shapely.geometry import box
from rasterio.enums import Resampling
import numpy as np
import matplotlib.pyplot as plt

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


def hillshade(array, azimuth=315, angle_altitude=45):
    # Converter ângulos para radianos
    az = np.deg2rad(azimuth)
    alt = np.deg2rad(angle_altitude)

    # Calcular gradientes
    x, y = np.gradient(array.astype(float))
    slope = np.pi/2. - np.arctan(np.sqrt(x*x + y*y))
    aspect = np.arctan2(-x, y)

    shaded = (np.sin(alt) * np.sin(slope) +
              np.cos(alt) * np.cos(slope) * np.cos(az - aspect))
    return 255 * (shaded + 1) / 2  # escala para 0-255


def resample_raster(src, new_resolution):
    scale = src.res[0] / new_resolution
    new_height = int(src.height / scale)
    new_width = int(src.width / scale)

    data_resampled = src.read(
        out_shape=(1, new_height, new_width),
        resampling=Resampling.bilinear
    )[0]

    return data_resampled



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
	
# rec_anadem = recortar_raster_por_coordenadas(anadem_path, minx_val, miny_val, maxx_val, maxy_val, "dados/rec_anadem.tif")
# rec_geosampa = recortar_raster_por_coordenadas(geosampa_path, minx_val, miny_val, maxx_val, maxy_val, "dados/rec_geosampa.tif")
# rec_anadem_teste = recortar_raster_por_coordenadas(
#     anadem_path, minx_val2, miny_val2, maxx_val2, maxy_val2, "dados/rec_anadem_teste.tif"
# )
# rec_geosampa_teste = recortar_raster_por_coordenadas(
#     geosampa_path, minx_val2, miny_val2, maxx_val2, maxy_val2, "dados/rec_geosampa_teste.tif"
# )

#    Recorte salvo em: dados/rec_anadem.tif
#    Dimensões: 195 x 130
#    Recorte salvo em: dados/rec_geosampa.tif
#    Dimensões: 11701 x 7801

src_path = os.path.join(data_dir, "rec_geosampa.tif")
with rasterio.open(src_path) as src:
    data = src.read(1)  # primeira banda
    profile = src.profile

with rasterio.open(src_path) as src:
    data_10m = resample_raster(src, 10)
    data_30m = resample_raster(src, 30)

# Gerar hillshades
hs_05 = hillshade(data)
hs_10 = hillshade(data_10m)
hs_30 = hillshade(data_30m)

# Plotar os três
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(hs_05, cmap='gray')
axes[0].set_title("Resolução 0.5 m")
axes[0].axis("off")

axes[1].imshow(hs_10, cmap='gray')
axes[1].set_title("Resolução 10 m")
axes[1].axis("off")

axes[2].imshow(hs_30, cmap='gray')
axes[2].set_title("Resolução 30 m")
axes[2].axis("off")

plt.tight_layout()
plt.show()