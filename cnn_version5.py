"""
1. python -m venv training
2. training\Scripts\activate (In base backend directory)
3. pip install h5py

Cuda environment setup:
1. cmd: nvidia-smi
2. Check what cuda version you would need to install (Right side).
3. Install: Correct CUDA Toolkit. (Example Toolkit: https://developer.nvidia.com/cuda-downloads)
4. Install: Correct torch version for your CUDA Toolkit within virtual environment from the website: https://pytorch.org/get-started/locally/ (Make sure to 'pip uninstall torch torchvision torchaudio' first)
Example command for synthura virtual environment: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Model attributes:
Epochs: 10
Batch Size: 32
Optimizer: Adam
Loss function: MSE
Activation functions: Leaky-relu + relu

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import os
from PIL import Image

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        attention_weights = torch.sigmoid(self.conv(x))
        return x * attention_weights

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.relu = nn.LeakyReLU(0.01, inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.avg_pool(x)
        attention = attention.view(attention.size(0), -1)
        attention = self.fc1(attention)
        attention = self.relu(attention)
        attention = self.fc2(attention)
        attention = self.sigmoid(attention)
        attention = attention.view(attention.size(0), attention.size(1), 1, 1)
        return x * attention

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dropout_prob):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout_prob)
        self.spatial_attention = SpatialAttention(out_channels)
        self.channel_attention = ChannelAttention(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.leaky_relu(x, negative_slope=0.01, inplace=True)
        x = self.dropout(x)
        x = self.spatial_attention(x)
        x = self.channel_attention(x)
        return x

class FCBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_prob):
        super(FCBlock, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x

class Network(nn.Module):
    def __init__(self, image_shape, dropout_probs):
        super(Network, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.conv1 = ConvBlock(3, 32, 5, 2, 2, dropout_probs[0]).to(self.device)
        self.conv2 = ConvBlock(32, 64, 3, 2, 1, dropout_probs[1]).to(self.device)
        self.conv3 = ConvBlock(64, 128, 3, 2, 1, dropout_probs[2]).to(self.device)
        self.conv4 = ConvBlock(128, 128, 3, 1, 1, dropout_probs[3]).to(self.device)
        self.conv5 = ConvBlock(128, 256, 3, 2, 1, dropout_probs[4]).to(self.device)
        self.conv6 = ConvBlock(256, 256, 3, 1, 1, dropout_probs[5]).to(self.device)
        self.conv7 = ConvBlock(256, 512, 3, 2, 1, dropout_probs[6]).to(self.device)
        self.conv8 = ConvBlock(512, 512, 3, 1, 1, dropout_probs[7]).to(self.device)
        self.conv9 = ConvBlock(512, 512, 3, 2, 1, dropout_probs[8]).to(self.device)
        self.conv10 = ConvBlock(512, 512, 3, 1, 1, dropout_probs[9]).to(self.device)

        conv_output_shape = self.calculate_conv_output_shape(image_shape)

        self.lstm = nn.LSTM(np.prod(conv_output_shape), 256, num_layers=2, batch_first=True).to(self.device)

        self.fc1 = FCBlock(256, 512, dropout_probs[10]).to(self.device)
        self.fc2 = FCBlock(512, 256, dropout_probs[11]).to(self.device)
        self.fc3 = FCBlock(256, 128, dropout_probs[12]).to(self.device)
        self.fc4 = FCBlock(128, 64, dropout_probs[13]).to(self.device)
        self.steer_output = nn.Linear(64, 1).to(self.device)

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)

    def calculate_conv_output_shape(self, input_shape):
        x = torch.zeros(1, 3, *input_shape).to(self.device)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        return x.shape[1:]

    def forward(self, x):
        x = x.to(self.device)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)

        x = torch.flatten(x, start_dim=1)
        x = x.unsqueeze(1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        steer = self.steer_output(x)

        return steer

def train(model, train_dataloader, val_dataloader, steer_criterion, optimizer, scheduler, num_epochs, l2_lambda):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    total_steps = len(train_dataloader)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")

        model.train()
        running_train_loss = 0.0
        for batch_idx, batch in enumerate(train_dataloader):
            images = batch['image'].to(device)
            steer_targets = batch['steer'].to(device).unsqueeze(1)

            optimizer.zero_grad()
            steer_output = model(images)
            steer_loss = steer_criterion(steer_output, steer_targets)

            l2_reg = torch.tensor(0.).to(device)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss = steer_loss + l2_lambda * l2_reg

            loss.backward()
            optimizer.step()

            running_train_loss += steer_loss.item()

            if (batch_idx + 1) % 1000 == 0 or (batch_idx + 1) == total_steps:
                print(f"  Batch [{batch_idx+1}/{total_steps}], Train Loss: {steer_loss.item():.4f}")

        epoch_train_loss = running_train_loss / total_steps

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                images = batch['image'].to(device)
                steer_targets = batch['steer'].to(device).unsqueeze(1)

                steer_output = model(images)
                steer_loss = steer_criterion(steer_output, steer_targets)

                running_val_loss += steer_loss.item()

        epoch_val_loss = running_val_loss / len(val_dataloader)

        scheduler.step(epoch_val_loss)

        print(f"\nEpoch [{epoch+1}/{num_epochs}] completed")
        print(f"  Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("Best model saved.")

    print("\nTraining completed!")
    print("Final model saved as 'model5.pth'")
    torch.save(model.state_dict(), 'model5.pth')

class CarlaDataset(data.Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, 'Images')
        self.steer_file = os.path.join(data_dir, 'SteerValues', 'steer_values.txt')

        self.image_files = sorted(os.listdir(self.image_dir))

        with open(self.steer_file, 'r') as file:
            self.steer_values = [float(line.strip()) for line in file]

        if len(self.image_files) != len(self.steer_values):
            raise ValueError("Number of image files and steer values do not match.")

        print(f"Number of image files: {len(self.image_files)}")
        print(f"Number of steer values: {len(self.steer_values)}")

    def __getitem__(self, index):
        image_file = self.image_files[index]
        steer_value = self.steer_values[index]

        image_path = os.path.join(self.image_dir, image_file)

        image = Image.open(image_path).convert('RGB')
        image = image.resize((200, 88))

        image = np.array(image)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        steer = torch.tensor(steer_value).float()

        return {'image': image, 'steer': steer}

    def __len__(self):
        return len(self.image_files)

if __name__ == '__main__':
    data_dir = 'archive/data'
    dataset = CarlaDataset(data_dir)

    print(f"Total dataset length: {len(dataset)}")

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = data.random_split(dataset, [train_size, val_size, test_size])

    train_dataloader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_dataloader = data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"\nTrain dataset length: {len(train_dataset)}")
    print(f"Validation dataset length: {len(val_dataset)}")
    print(f"Test dataset length: {len(test_dataset)}")

    image_shape = (88, 200)

    dropout_probs = [0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5]
    model = Network(image_shape, dropout_probs)
    steer_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00005, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    num_epochs = 10
    l2_lambda = 0.001

    print("\nGPU available:", torch.cuda.is_available())
    print("\nStarting training...")

    train(model, train_dataloader, val_dataloader, steer_criterion, optimizer, scheduler, num_epochs, l2_lambda)