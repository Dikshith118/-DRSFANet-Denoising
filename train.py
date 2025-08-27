import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import os
import argparse

import config
from model import DRSFANet
from dataset import DenoisingDataset
from utils import calculate_psnr, calculate_ssim

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def check_accuracy(loader, model, device="cuda"):
    model.eval()
    total_psnr = 0
    total_ssim = 0
    num_samples = 0

    with torch.no_grad():
        for noisy, clean in loader:
            noisy = noisy.to(device)
            clean = clean.to(device)
            
            # The model now returns the predicted noise map
            predicted_noise = model(noisy)
            
            # Denoised image is manually calculated for accuracy check
            denoised = noisy - predicted_noise

            total_psnr += calculate_psnr(clean, denoised) * noisy.size(0)
            total_ssim += calculate_ssim(clean, denoised) * noisy.size(0)
            num_samples += noisy.size(0)

    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples
    print(f"Validation PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.4f}")
    model.train()
    return avg_psnr

def train_fn(loader, model, optimizer, loss_fn):
    model.train()

    for batch_idx, (noisy, clean) in enumerate(loader):
        noisy = noisy.to(config.DEVICE)
        clean = clean.to(config.DEVICE)

        # The model's output is the predicted noise map
        predicted_noise = model(noisy)
        
        # Calculate the target noise map
        target_noise = noisy - clean
        
        # The loss is between the predicted noise and the target noise
        loss = loss_fn(predicted_noise, target_noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}")

def main():
    parser = argparse.ArgumentParser(description="Train DRSFANet model.")
    parser.add_argument('--model_type', type=str, default='color', choices=['grayscale', 'color'], help="Type of model to train ('grayscale' or 'color').")
    args = parser.parse_args()

    in_channels = 1 if args.model_type == 'grayscale' else 3
    
    checkpoint_filename = f"drsfanet_{args.model_type}_best.pth.tar"
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, checkpoint_filename)

    print("--- Starting Training ---")
    print(f"Device: {config.DEVICE}")
    print(f"Training {args.model_type} model with {in_channels} channels.")

    train_dataset = DenoisingDataset(root_dir=config.TRAIN_DIR, model_type=args.model_type)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    val_dataset = DenoisingDataset(root_dir=config.TRAIN_DIR, model_type=args.model_type) 
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    model = DRSFANet(in_channels=in_channels).to(config.DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    
    best_psnr = 0.0

    for epoch in range(config.NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{config.NUM_EPOCHS} ---")
        train_fn(train_loader, model, optimizer, loss_fn)
        
        scheduler.step()
        
        current_psnr = check_accuracy(val_loader, model, device=config.DEVICE)

        if current_psnr > best_psnr:
            best_psnr = current_psnr
            save_checkpoint(model, optimizer, filename=checkpoint_path)
            
    print("--- Training complete! ---")

if __name__ == "__main__":
    main()