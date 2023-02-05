import torch
from PIL import Image

LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_GEN = "gen.pth"
CHECKPOINT_DISC = "disc.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10000
BATCH_SIZE = 16
LAMBDA_GP = 10
NUM_WORKERS = 4
HIGH_RES = 128
LOW_RES = HIGH_RES // 4
IMG_CHANNELS = 3