import os
import argparse

"""
CLI arguments for encoder pre-training
"""

parser = argparse.ArgumentParser()

# for encoder_pre-training.py
parser.add_argument('--GPU', type=int, default=0, help='GPU number to use')
parser.add_argument('--LOAD_PATH', type=str, default='/scratch/zceerba/projectSERN/audio_hr_v2/checkpoints/best_model.pth', help='Path to the best pre-trained encoder model')

config = parser.parse_args()
config.DEVICE = 'cuda' if config.GPU >= 0 else 'cpu'