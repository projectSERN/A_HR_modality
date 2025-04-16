import os
import argparse

"""
CLI arguments for encoder pre-training
"""

parser = argparse.ArgumentParser()

# for encoder_pre-training.py
parser.add_argument('--GPU', type=int, default=0, help='GPU number to use')
parser.add_argument('--LOAD_PATH', type=str, default='/scratch/zceerba/projectSERN/audio_hr_v2/checkpoints/best_model.pth', help='Path to the best pre-trained encoder model')
parser.add_argument('-BS', '--BATCH_SIZE', type=int, default=32, help='Batch size for training')
parser.add_argument('--EPOCHS', type=int, default=100, help='Number of epochs for training')
parser.add_argument('-LR', '--LEARNING_RATE', type=float, default=0.0001, help='Learning rate for the optimizer')
parser.add_argument('-L2', '--LAMBDA2', type=float, default=0.01, help='L2 regularization weight')
parser.add_argument('-D','--DROPOUT', type=float, default=0.5, help='Dropout rate for the model')
parser.add_argument('-H', '--HIDDEN_SIZE', type=int, default=128, help='Hidden size for the model')
parser.add_argument('-NL', '--NUM_LAYERS', type=int, default=2, help='Number of layers for the model')
parser.add_argument('--MODEL', type=str, default='lstm', choices=['lstm', 'conv'], help='Model name to use for training')
parser.add_argument('-KS', '--KERNEL_SIZE', type=int, default=3, help='Kernel size for the convolutional model')
parser.add_argument('--PADDING', type=int, default=1, help='Padding for the convolutional model')

# for base_model_training.py
parser.add_argument('--SUBSET', type=bool, default=False, help='Use a subset of the data for training')
parser.add_argument('--CLIP_LENGTH', type=int, default=10, help='Length of the clips for training')

config = parser.parse_args()
config.DEVICE = 'cuda' if config.GPU >= 0 else 'cpu'