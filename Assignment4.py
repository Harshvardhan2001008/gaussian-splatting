import os
import numpy as np
import imageio # Or use Pillow (PIL)
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage import io # scikit-image's own I/O is also an option

# --- Configuration ---
# Path to the folder containing your ORIGINAL Apollo 17 images
original_images_folder = 'output/000fab2b-8/train/ours_30000/gt' # <<< MAKE SURE THIS PATH IS CORRECT

# Path to the folder containing the images RENDERED by render.py
# Based on the render.py script and your previous output, this is likely:
# <your_model_path>/test/ours_<loaded_iteration_number>/renders/
# Replace with the actual path from your render run
rendered_images_folder = 'output/000fab2b-8/train/ours_30000/renders' # <<< MAKE SURE THIS PATH IS CORRECT (replace <ITER_NUMBER>)

# Get a list of image filenames from the rendered folder
# We assume the filenames in the rendered folder match the original ones or have a consistent mapping
# The render.py script saves as 00000.png, 00001.png etc.
# You'll need to map these back to the original image names for comparison,
# or just ensure the list order corresponds correctly if the rendered images are sorted.
# A more robust way is to list original images and find the corresponding rendered file if name matches or index matches.

# Let's assume for now that the rendered files are named 00000.png, 00001.png, ..., 00014.png
# and correspond to the original images when sorted alphabetically or by index in the render list.

# --- Get lists of files ---
# List files in the rendered folder. Render.py saves as .png.
rendered_files = sorted([f for f in os.listdir(rendered_images_folder) if f.endswith('.png')])
print(f"Found {len(rendered_files)} rendered images.")

# You need to get the list of original files that correspond to these.
# If render.py saved them in the same order as the original COLMAP list,
# you can just list and sort the original image files.
original_files_all = sorted([f for f in os.listdir(original_images_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

# Find the original files that were part of the 15 used for training/testing
# This requires knowing which 15 specific files were used.
# If all files in 'data/apollo17/input' are the 15 you used, use original_files_all[:15] (assuming sorted order matches COLMAP's internal list)
# A more robust approach would be to get the list of image names from the COLMAP images.txt or the scene object if possible.
# For this example, let's assume the first 15 sorted original images correspond to the rendered 00000.png to 00014.png
original_files = original_files_all[:len(rendered_files)] # Get the first N original files corresponding to rendered ones
print(f"Comparing against {len(original_files)} original images.")


# --- Calculate Metrics ---
psnr_values = []
ssim_values = []

if len(original_files) != len(rendered_files) or not original_files:
    print("Error: Number of original and rendered images do not match, or no files found.")
else:
    print("Calculating PSNR and SSIM...")
    for i in range(len(rendered_files)):
        original_filepath = os.path.join(original_images_folder, original_files[i])
        rendered_filepath = os.path.join(rendered_images_folder, rendered_files[i])

        try:
            # Load images using scikit-image's io or imageio
            # Using io.imread usually loads as uint8, need to convert to float 0-1 for metrics
            # Using imageio.imread can sometimes load as float directly depending on file type and version
            # Let's use imageio and ensure float [0, 1]
            img_original = imageio.imread(original_filepath).astype(np.float32) / 255.0
            img_rendered = imageio.imread(rendered_filepath).astype(np.float32) / 255.0

            # Ensure images have the same dimensions (sanity check)
            if img_original.shape != img_rendered.shape:
                 print(f"Warning: Image dimensions mismatch for {original_files[i]} vs {rendered_files[i]}. Skipping.")
                 continue

            # Calculate PSNR
            # data_range is the maximum possible value of the input data type (1.0 for float [0,1], 255 for uint8)
            psnr_val = peak_signal_noise_ratio(img_original, img_rendered, data_range=1.0)
            psnr_values.append(psnr_val)

            # Calculate SSIM
            # channel_axis=2 is needed for color images (RGB)
            # data_range is max value (1.0 for float [0,1])
            # Use channel_axis=None if images are grayscale
            ssim_val = structural_similarity(img_original, img_rendered, data_range=1.0, channel_axis=2)
            ssim_values.append(ssim_val)

            print(f"  {original_files[i]}: PSNR={psnr_val:.4f}, SSIM={ssim_val:.4f}")

        except FileNotFoundError:
            print(f"Error: File not found - {original_filepath} or {rendered_filepath}. Skipping.")
        except Exception as e:
            print(f"Error processing {original_files[i]} and {rendered_files[i]}: {e}. Skipping.")


# --- Print Average Metrics ---
if psnr_values and ssim_values:
    average_psnr = np.mean(psnr_values)
    average_ssim = np.mean(ssim_values)
    print("\n--- Average Metrics ---")
    print(f"Average PSNR over {len(psnr_values)} images: {average_psnr:.4f}")
    print(f"Average SSIM over {len(ssim_values)} images: {average_ssim:.4f}")
else:
    print("\nNo metrics calculated.")
