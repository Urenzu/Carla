import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import os
from PIL import Image
from torchvision.models import vit_b_16, ViT_B_16_Weights

class SpatialAttention(nn.Module):

class ChannelAttention(nn.Module):

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, qkv_bias=False, dropout_prob=0.0):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_prob, bias=qkv_bias)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout_prob)
        )

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        mlp_output = self.mlp(x)
        x = x + mlp_output
        x = self.norm2(x)
        return x

class Network(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super(Network, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        self.feature_extractor = vit.encoder
        
        self.spatial_attention = SpatialAttention(768)
        self.channel_attention = ChannelAttention(768)

        self.fc1 = nn.Linear(768, 256)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc3 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(dropout_prob)
        self.steer_output = nn.Linear(64, 1)

    def forward(self, x):
        x = x.to(self.device)

        x = self.feature_extractor(x)
        x = x.last_hidden_state
        x = x[:, 0]  # Take the CLS token output
        x = self.spatial_attention(x)
        x = self.channel_attention(x)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        steer = self.steer_output(x)

        return steer
