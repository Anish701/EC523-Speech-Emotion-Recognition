import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Base_CNN(nn.Module):
    def __init__(self):
        super(Base_CNN, self).__init__()
        
        #declare layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.max_pool = nn.AdaptiveMaxPool2d((1,1))
        
        self.dropout = nn.Dropout(0.3)
        
        self.classifier1 = nn.Linear(256, 64)
        self.classifier2 = nn.Linear(64, 6)
        
        self.residualConv = nn.Conv2d(1, 64, kernel_size=1, stride=2, padding=1)
        
    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        
        residual = self.residualConv(residual)
        
        x = F.pad(x, (0, 1))
        
        residual = residual[:, :, :x.shape[2], :x.shape[3]]
        
        # print(f"x shape: {x.shape}, res shape: {residual.shape}")
        
        x = x + residual
        
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)        

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool(x)
        
        #to fix dimensionality
        x = self.max_pool(x)
        x = torch.flatten(x, 1)
        
        x = self.dropout(x)
        
        x = self.classifier1(x)
        x = self.classifier2(x)

        return x

class Base_CNN_Simplified(nn.Module):
    def __init__(self):
        super(Base_CNN_Simplified, self).__init__()
        
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        self.max_pool = nn.AdaptiveMaxPool2d((1,1))
        self.classifier = nn.Linear(64, 6)

    def forward(self, x):
        x = self.relu(self.pool(self.conv1(x)))
        x = self.relu(self.pool(self.conv2(x)))

        x = self.max_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return x

class Base_CNN_Transformer(nn.Module):
    def __init__(self, transformer_layers=2, n_heads=4, transformer_dim=256, input_freq_bins=8):
        super(Base_CNN_Transformer, self).__init__()

        # CNN layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, transformer_dim, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(transformer_dim)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        # Projection from D*F to D
        self.project = nn.Linear(transformer_dim * input_freq_bins, transformer_dim)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=n_heads, dropout=0.2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.pos_encoder = PositionalEncoding(transformer_dim)

        # Classification
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(transformer_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
        )

    def apply_layernorm(self, x):
        B, C, H, W = x.shape
        return nn.LayerNorm([C, H, W]).to(x.device)(x)

    def forward(self, x):
        # CNN
        x = self.relu(self.apply_layernorm(self.pool(self.conv1(x))))
        x = self.relu(self.apply_layernorm(self.pool(self.conv2(x))))
        x = self.relu(self.apply_layernorm(self.pool(self.conv3(x))))
        x = self.relu(self.apply_layernorm(self.pool(self.conv4(x))))  # [B, D, F, T]

        B, D, F, T = x.shape

        # Rearrange for transformer: each time step is a token
        x = x.permute(0, 3, 1, 2)         # [B, T, D, F]
        x = x.reshape(B, T, D * F)        # [B, T, D*F]
        x = self.project(x)               # [B, T, D]

        # Transformer expects [T, B, D]
        x = x.permute(1, 0, 2)            # [T, B, D]
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)

        # Back to [B, D, T] for pooling
        x = x.permute(1, 2, 0)            # [B, D, T]
        x = self.global_pool(x).squeeze(2)  # [B, D]

        x = self.classifier(x)  # [B, 6]
        return x

class Base_CNN_GRU(nn.Module):
    def __init__(self):
        super(Base_CNN_GRU, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        self.residualConv = nn.Conv2d(1, 64, kernel_size=1, stride=2, padding=1)

        # GRU expects input_size = 256 (channels), and sequence length = width
        self.gru = nn.GRU(input_size=256*8, hidden_size=128, num_layers=1,
                          batch_first=True, bidirectional=True)

        self.classifier1 = nn.Linear(128 * 2, 64)  # bidirectional
        self.classifier2 = nn.Linear(64, 6)

    def forward(self, x):
        residual = x  # x: [B, 1, 128, 256]

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.bn1(x)

        x = self.conv2(x)
        residual = self.residualConv(residual)
        x = F.pad(x, (0, 1))  # pad width to align
        residual = residual[:, :, :x.shape[2], :x.shape[3]]
        x = x + residual

        x = self.relu(x)
        x = self.pool(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.bn3(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.bn4(x)

        # x: [B, 256, H, W] — after all CNN and pooling layers
        B, C, H, W = x.shape
        
        # Reshape for GRU: treat W as time steps, and C*H as input features
        x = x.permute(0, 3, 1, 2)  # [B, W, C, H]
        x = x.contiguous().view(B, W, C * H)  # [B, W, C*H]

        # Update GRU input size if needed
        x, _ = self.gru(x)  # GRU input_size = C*H

        x = x[:, -1, :]  # last time step
        x = self.classifier1(x)
        x = self.classifier2(x)

        return x