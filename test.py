import torch
import os
from PIL import Image
import torchvision.transforms as transforms
import argparse
import glob
import numpy as np

# Import our project components
import config
from model import DRSFANet
from utils import calculate_psnr, calculate_ssim

def load_model(model_path, model_type):
    """Loads the saved model checkpoint."""
    print(f"=> Loading checkpoint: {model_path}")
    
    in_channels = 1 if model_type == 'grayscale' else 3
    model = DRSFANet(in_channels=in_channels).to(config.DEVICE)
    
    checkpoint = torch.load(model_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model

def process_single_image(model, image_path, model_type, save_dir):
    """
    Denoises a single image, calculates metrics, and saves the result in a specified directory.
    Returns the PSNR and SSIM values.
    """
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path)
    
    if model_type == 'grayscale':
        image = image.convert('L')
    else:
        image = image.convert('RGB')
        
    original_tensor = transform(image).unsqueeze(0).to(config.DEVICE)
    
    noise_level = 25 / 255.0
    noise = torch.randn_like(original_tensor) * noise_level
    noisy_tensor = torch.clamp(original_tensor + noise, 0., 1.)
    
    with torch.no_grad():
        denoised_tensor = model(noisy_tensor)
        
    psnr_score = calculate_psnr(original_tensor, denoised_tensor)
    ssim_score = calculate_ssim(original_tensor, denoised_tensor)
    
    print(f"Processing {os.path.basename(image_path)} -> PSNR: {psnr_score:.2f} | SSIM: {ssim_score:.4f}")
    
    # --- New logic for saving to a subfolder ---
    output_image = transforms.ToPILImage()(denoised_tensor.squeeze(0).cpu())
    base_filename = os.path.basename(image_path)
    # Ensure the new subdirectory exists
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"denoised_{base_filename}")
    output_image.save(save_path)
    
    return psnr_score, ssim_score

def main():
    parser = argparse.ArgumentParser(description="Test DRSFANet on a single image or a dataset.")
    parser.add_argument('--model_type', type=str, required=True, choices=['grayscale', 'color'], help="Type of model to use ('grayscale' or 'color').")
    parser.add_argument('--path', type=str, required=True, help="Path to the image or directory you want to denoise.")
    args = parser.parse_args()

    if args.model_type == 'grayscale':
        model_filename = "drsfanet_grayscale_best.pth"
    else:
        model_filename = "drsfanet_color_best.pth"
        
    model_path = os.path.join(config.CHECKPOINT_DIR, model_filename)

    model = load_model(model_path, args.model_type)
    
    psnr_scores = []
    ssim_scores = []
    
    if os.path.isdir(args.path):
        image_files = glob.glob(os.path.join(args.path, "*"))
        if not image_files:
            print("No image files found in the specified directory.")
            return

        # --- New logic for the save directory ---
        dataset_name = os.path.basename(args.path)
        save_dir = os.path.join(config.RESULTS_DIR, dataset_name)
        
        for image_file in image_files:
            try:
                psnr_val, ssim_val = process_single_image(model, image_file, args.model_type, save_dir)
                psnr_scores.append(psnr_val)
                ssim_scores.append(ssim_val)
            except Exception as e:
                print(f"Skipping {image_file} due to error: {e}")
            
        if psnr_scores:
            avg_psnr = np.mean(psnr_scores)
            avg_ssim = np.mean(ssim_scores)
            print("\n" + "="*50)
            print(f"Testing complete on {len(image_files)} images!")
            print(f"Average PSNR: {avg_psnr:.2f} | Average SSIM: {avg_ssim:.4f}")
            print("="*50)
        else:
            print("\nNo images were successfully processed for PSNR/SSIM calculation.")
        
    elif os.path.isfile(args.path):
        # For a single image, save it in the default results directory
        save_dir = config.RESULTS_DIR
        process_single_image(model, args.path, args.model_type, save_dir)
        
    else:
        print("Error: The provided path is not a valid file or directory.")

if __name__ == "__main__":
    main()