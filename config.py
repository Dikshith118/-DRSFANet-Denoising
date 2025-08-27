import torch

# --- Training Hyperparameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-3     # The value from the paper
BATCH_SIZE = 32          # The value from the paper
NUM_EPOCHS = 30        # Train for more epochs to ensure convergence

# --- Project Paths ---
# The correct folder names for your project structure
TRAIN_DIR = "data/train/BSD400"
VAL_DIR = "data/val/"
CHECKPOINT_DIR = "checkpoints/"
RESULTS_DIR = "results/"

# --- Model Saving & Loading ---
SAVE_CHECKPOINT = True
LOAD_MODEL = False
CHECKPOINT_NAME = "drsfanet_color_best.pth"