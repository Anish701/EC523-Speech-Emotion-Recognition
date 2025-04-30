import librosa
import numpy as np
import torch

def extract_mfcc(path):
    audio, sr = librosa.load(path, sr=22000, duration=4, mono=True)
    if len(audio) < 4 * sr:
        audio = np.pad(audio, pad_width=(0, 4 * sr - len(audio)), mode='constant')

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=128)
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-6)

    return torch.tensor(mfcc, dtype=torch.float32)

label_map = {
    0: 'anger',
    1: 'disgust',
    2: 'fear',
    3: 'happiness',
    4: 'neutral',
    5: 'sadness'
}
