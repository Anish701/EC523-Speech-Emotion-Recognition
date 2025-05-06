import numpy as np
import librosa
import torch
from torch.utils.data import Dataset

def process_audio(path):
    ''' 
    Load the audio file, convert the audio file into a mel spectrogram,
    return the mel spectrogram as an image, and convert the image into a numpy array
    '''
    audio, sr = librosa.load(path, sr=22000, duration=4, mono=True)
    if len(audio) < 4 * sr:
        audio = np.pad(audio, pad_width=(0, 4 * sr - len(audio)), mode='constant')
    
    signal = librosa.feature.melspectrogram(y = audio, sr=sr, n_mels=128)

    signal = librosa.power_to_db(signal, ref=np.min)    
    image = torch.tensor(signal, dtype=torch.float32)
    
    image = (image - image.mean()) / (image.std() + 1e-6)
    
    return image

def extract_mfcc(path):
    ''' 
    Load the audio file, convert the audio file into MFCCs and return the MFCCs
    '''
    audio, sr = librosa.load(path, sr=22000, duration=4, mono=True)
    if len(audio) < 4 * sr:
        audio = np.pad(audio, pad_width=(0, 4 * sr - len(audio)), mode='constant')

    signal = librosa.feature.mfcc(y = audio, sr=sr, n_mfcc=128)
    signal = (signal - signal.mean()) / (signal.std() + 1e-6)
    
    return torch.tensor(signal, dtype=torch.float32)

def collate_fn(batch):
    spectrograms, labels = zip(*batch)
    
    max_length = max(spec.shape[1] for spec in spectrograms)

    spectrograms_padded = [torch.nn.functional.pad(spec, (0, max_length - spec.shape[1])) for spec in spectrograms]

    spectrograms_padded = torch.stack(spectrograms_padded)

    labels = torch.tensor(labels, dtype=torch.long)
    return spectrograms_padded, labels


class AudioDataset(Dataset):
    def __init__(self, dataframe, augment=False):
        self.df = dataframe
        self.labels = [torch.tensor(label, dtype=torch.long) for label in self.df['Emotion']]
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['Path']
        label = row['Emotion']

        if not self.augment:
            spectrogram = extract_mfcc(path)
        else:
            audio, sr = librosa.load(path, sr=22000, duration=4, mono=True)

            if len(audio) < 4 * sr:
                audio = np.pad(audio, pad_width=(0, 4 * sr - len(audio)), mode='constant')

            if self.augment:
                audio = self.apply_augmentation(audio, sr)

            signal = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=128)
            signal = (signal - signal.mean()) / (signal.std() + 1e-6)
            spectrogram = torch.tensor(signal, dtype=torch.float32)

        return spectrogram, torch.tensor(label, dtype=torch.long)
    
    def apply_augmentation(self, audio, sr):
        if random.random() < 0.5:
            audio = librosa.effects.time_stretch(audio, rate=random.uniform(0.9, 1.1))
        if random.random() < 0.5:
            audio = librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=random.choice([-2, -1, 1, 2]))
        if random.random() < 0.5:
            audio += 0.005 * np.random.randn(len(audio))
        if random.random() < 0.5:
            audio = audio * random.uniform(0.8, 1.2)
        return audio