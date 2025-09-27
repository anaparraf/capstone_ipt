# import torch

# print("Torch version:", torch.__version__)
# print("CUDA available:", torch.cuda.is_available())
# if torch.cuda.is_available():
#     print("GPU:", torch.cuda.get_device_name(0))


from unet import generate_super_resolution

# Set your paths and parameters
model_path = "adaptive_unet.pth"
input_raster_path = "dados/rec_anadem.tif"
output_path = "output/anadem_16f_40ep_10m.tif"
target_resolution = 10

generate_super_resolution(
    model_path,
    input_raster_path,
    output_path,
    target_resolution=target_resolution
)
