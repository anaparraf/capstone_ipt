import rasterio
import os
from rasterio.mask import mask
from shapely.geometry import box

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
