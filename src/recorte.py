# ...existing code...
import os
import math
import rasterio
from rasterio import windows
from rasterio.warp import transform_bounds, reproject, Resampling
from rasterio.transform import from_bounds
from shapely.geometry import box

def print_source_info(src_fp, dst_crs="EPSG:31983"):
    if not os.path.exists(src_fp):
        raise SystemExit(f"Source not found: {src_fp}")
    with rasterio.open(src_fp) as src:
        print("SRC PATH:", src_fp)
        print("SRC CRS:", src.crs)
        print("SRC width,height:", src.width, src.height)
        print("SRC transform:", src.transform)
        print("SRC bounds (src CRS):", src.bounds)
        tb = transform_bounds(src.crs, dst_crs, *src.bounds, densify_pts=21)
        print(f"SRC bounds in {dst_crs}:", tb)
        minx, miny, maxx, maxy = tb
        cx = (minx + maxx) / 2.0
        cy = (miny + maxy) / 2.0
        print("Center in", dst_crs, ":", (cx, cy))
        return tb, (cx, cy)

def suggest_centered_bbox(tb_dst, size_m=2000):
    minx, miny, maxx, maxy = tb_dst
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0
    half = size_m / 2.0
    xmin = max(minx, cx - half)
    xmax = min(maxx, cx + half)
    ymin = max(miny, cy - half)
    ymax = min(maxy, cy + half)
    if (xmax - xmin) < size_m:
        diff = size_m - (xmax - xmin)
        xmin = max(minx, xmin - diff/2.0)
        xmax = min(maxx, xmax + diff/2.0)
    if (ymax - ymin) < size_m:
        diff = size_m - (ymax - ymin)
        ymin = max(miny, ymin - diff/2.0)
        ymax = min(maxy, ymax + diff/2.0)
    return (float(xmin), float(ymin), float(xmax), float(ymax))

def crop_reproject_window(src_fp,
                          dst_fp,
                          desired_bbox_31983,
                          other_tif_paths=None,
                          dst_crs="EPSG:31983",
                          dst_res=5,
                          resampling=Resampling.bilinear,
                          num_threads=4):
    other_tif_paths = other_tif_paths or []
    desired_box = box(*desired_bbox_31983)

    # check intersection with other tifs (transform their bounds to dst_crs)
    for ot in other_tif_paths:
        if not os.path.exists(ot):
            continue
        with rasterio.open(ot) as other:
            other_bounds_31983 = transform_bounds(other.crs, dst_crs, *other.bounds, densify_pts=21)
            other_box = box(*other_bounds_31983)
            if desired_box.intersects(other_box):
                raise SystemExit(f"Desired bbox intersects other tif: {ot}")

    if not os.path.exists(src_fp):
        raise SystemExit(f"Source file not found: {src_fp}")

    with rasterio.open(src_fp) as src:
        src_crs = src.crs
        # transform desired bbox (31983) to source CRS
        minx_s, miny_s, maxx_s, maxy_s = transform_bounds(dst_crs, src_crs, *desired_bbox_31983, densify_pts=21)

        # build pixel window on source
        win = windows.from_bounds(minx_s, miny_s, maxx_s, maxy_s, src.transform)

        col_off = int(math.floor(win.col_off))
        row_off = int(math.floor(win.row_off))
        width = int(math.ceil(win.width + (win.col_off - col_off)))
        height = int(math.ceil(win.height + (win.row_off - row_off)))

        col_off = max(0, col_off)
        row_off = max(0, row_off)
        width = min(width, src.width - col_off)
        height = min(height, src.height - row_off)

        if width <= 0 or height <= 0:
            raise SystemExit("A janela calculada está fora do raster fonte.")

        win = windows.Window(col_off, row_off, width, height)
        src_win_bounds = windows.bounds(win, src.transform)

        dst_minx, dst_miny, dst_maxx, dst_maxy = transform_bounds(src_crs, dst_crs, *src_win_bounds, densify_pts=21)
        dst_width = int(math.ceil((dst_maxx - dst_minx) / dst_res))
        dst_height = int(math.ceil((dst_maxy - dst_miny) / dst_res))
        if dst_width <= 0 or dst_height <= 0:
            raise SystemExit("Dimensões destino inválidas. Verifique bbox e resolução.")

        dst_transform = from_bounds(dst_minx, dst_miny, dst_maxx, dst_maxy, dst_width, dst_height)

        profile = src.profile.copy()
        profile.update({
            "crs": dst_crs,
            "transform": dst_transform,
            "width": dst_width,
            "height": dst_height,
            "compress": "LZW",
            "tiled": False,
            "bigtiff": "YES",
            "driver": "GTiff",
            "count": src.count,
            "dtype": src.dtypes[0]
        })

        os.makedirs(os.path.dirname(dst_fp), exist_ok=True)
        print(f"Recortando janela (fonte pixels): col_off={col_off}, row_off={row_off}, width={width}, height={height}")
        print(f"Bounds fonte (CRS fonte): {src_win_bounds}")
        print(f"Destino: {dst_width}x{dst_height}, transform: {dst_transform}")

        with rasterio.open(dst_fp, "w", **profile) as dst:
            for b in range(1, src.count + 1):
                arr = src.read(b, window=win)
                src_win_transform = windows.transform(win, src.transform)
                reproject(
                    source=arr,
                    destination=rasterio.band(dst, b),
                    src_transform=src_win_transform,
                    src_crs=src_crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=resampling,
                    num_threads=num_threads
                )
    return dst_fp

if __name__ == "__main__":
    # adjust these paths if needed
    src_path = r"D:\git\capstone_ipt\saopaulo_menos_teste.tif"
    dst_path = r"D:\git\capstone_ipt\dados\sp_5m_tile.tif"
    other_tifs = [r"D:\git\capstone_ipt\output\anadem_u_net2.tif"]  # don't overlap these

    # inspect source and get suggested bbox (centered 2000m box)
    tb_dst, center = print_source_info(src_path, dst_crs="EPSG:31983")
    suggested = suggest_centered_bbox(tb_dst, size_m=2000)
    print("\nUsing suggested bbox (EPSG:31983):", suggested)

    # if you want a different size, change size_m above or replace `suggested` with your bbox
    out = crop_reproject_window(src_path, dst_path, suggested, other_tif_paths=other_tifs, dst_res=5, num_threads=4)
    print("Arquivo gerado:", out)
# ...existing code...