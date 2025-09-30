import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

def resample_to_match(src, reference_path):
    """Reprojeta/reamostra raster `src` para ter mesmo grid do raster de referência."""
    with rasterio.open(reference_path) as ref:
        transform = ref.transform
        width = ref.width
        height = ref.height
        crs = ref.crs
        dtype = src.profile["dtype"]

        profile = ref.profile.copy()
        profile.update({"dtype": dtype})

        data_resampled = np.empty((height, width), dtype=dtype)

        reproject(
            source=rasterio.band(src, 1),
            destination=data_resampled,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=crs,
            resampling=Resampling.bilinear
        )

    return data_resampled, profile

    data_resampled = np.empty((height, width), dtype=src.profile["dtype"])

    reproject(
        source=rasterio.band(src, 1),
        destination=data_resampled,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=transform,
        dst_crs=target_crs,
        resampling=Resampling.bilinear
    )

    return data_resampled, profile


def compare_rasters(path1, path2):
    """Compara dois rasters alinhados, ignorando NODATA."""
    with rasterio.open(path1) as src1, rasterio.open(path2) as src2:
        arr1 = src1.read(1).astype(np.float32)
        arr2 = src2.read(1).astype(np.float32)

    nodata = -3.4028235e+38
    mask = (arr1 != nodata) & (arr2 != nodata)

    diff = np.zeros_like(arr1, dtype=np.float32)
    diff[mask] = arr1[mask] - arr2[mask]
    diff_abs = np.abs(diff[mask])

    print("\n=== Estatísticas de diferença (30 m, SIRGAS2000) ===")
    print(f"Pixels válidos: {diff_abs.size}")
    print(f"Média da diferença: {np.mean(diff_abs):.3f}")
    print(f"Desvio-padrão: {np.std(diff_abs):.3f}")
    print(f"Min diff: {np.min(diff_abs):.3f}")
    print(f"Max diff: {np.max(diff_abs):.3f}")


geosampa_path = "dados/rec_geosampa.tif"
anadem_path   = "dados/rec_anadem.tif"

# Reprojetar GeoSampa para o grid do ANaDEM
with rasterio.open(geosampa_path) as src:
    data_geo, profile_geo = resample_to_match(src, anadem_path)

    with rasterio.open("geosampa_match_anadem.tif", "w", **profile_geo) as dst:
        dst.write(data_geo, 1)

# Agora comparar (grades iguais!)
compare_rasters("geosampa_match_anadem.tif", anadem_path)


# === Estatísticas de diferença (30 m, SIRGAS2000) ===
# Pixels válidos: 25325
# Média da diferença: 2.477
# Desvio-padrão: 1.519
# Min diff: 0.001
# Max diff: 23.532