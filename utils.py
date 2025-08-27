import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim

def calculate_psnr(original, denoised):
    """
    Calculates the Peak Signal-to-Noise Ratio (PSNR) between two images.
    Images are expected to be PyTorch tensors with values in the [0, 1] range.
    """
    # This function works correctly for batches, as torch.mean averages over all elements.
    mse = torch.mean((original - denoised) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()

def calculate_ssim(original, denoised):
    """
    Calculates the Structural Similarity Index (SSIM) for a batch of images.
    Handles the conversion from PyTorch tensors (B, C, H, W) to the format
    expected by scikit-image (H, W, C).
    """
    ssim_sum = 0.0
    batch_size = original.size(0)

    # Loop through each image in the batch
    for i in range(batch_size):
        # Get single images from the batch
        original_img_tensor = original[i]
        denoised_img_tensor = denoised[i]

        # Convert single tensors to NumPy arrays
        original_np = original_img_tensor.cpu().detach().numpy()
        denoised_np = denoised_img_tensor.cpu().detach().numpy()

        # Transpose from (C, H, W) to (H, W, C) for scikit-image
        original_np = np.transpose(original_np, (1, 2, 0))
        denoised_np = np.transpose(denoised_np, (1, 2, 0))

        # --- THIS IS THE FIX ---
        # Calculate SSIM for the single color image, specifying the channel axis
        ssim_score = ssim(original_np, denoised_np, data_range=1.0, channel_axis=-1)
        ssim_sum += ssim_score

    # Return the average SSIM for the batch
    return ssim_sum / batch_size
