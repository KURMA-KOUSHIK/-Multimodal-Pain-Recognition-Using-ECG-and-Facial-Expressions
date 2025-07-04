import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18
from torch.utils.data import Dataset, DataLoader

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transform for video frames
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# FacialVideoEncoder model
class FacialVideoEncoder(nn.Module):
    def __init__(self, num_classes=4):
        super(FacialVideoEncoder, self).__init__()
        self.feature_extractor = resnet18(pretrained=True)
        self.feature_extractor.fc = nn.Identity()
        self.temporal_encoder = nn.GRU(input_size=512, hidden_size=256, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(256 * 2, num_classes)

    def forward(self, x):
        batch_size, num_frames, _, _, _ = x.size()
        x = x.view(batch_size * num_frames, 3, 224, 224)
        features = self.feature_extractor(x)
        features = features.view(batch_size, num_frames, -1)
        _, hidden = self.temporal_encoder(features)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        out = self.fc(hidden)
        return out

# HeartRateEncoder model
class HeartRateEncoder(nn.Module):
    def __init__(self, input_dim=1, embed_dim=512, num_classes=4):
        super(HeartRateEncoder, self).__init__()
        self.input_projection = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.output_layer = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.output_layer(x)
        return x

# Combined dataset class
class CombinedBioVidDataset(Dataset):
    def __init__(self, video_base_folder, ecg_base_folder, max_frames=138, max_length=100):
        self.video_base_folder = video_base_folder
        self.ecg_base_folder = ecg_base_folder
        self.max_frames = max_frames
        self.max_length = max_length
        self.subjects = sorted(os.listdir(video_base_folder))
        self.valid_data = self._load_files()

    def _load_files(self):
        valid_data = []
        for subject_folder in self.subjects:
            video_folder_path = os.path.join(self.video_base_folder, subject_folder)
            ecg_folder_path = os.path.join(self.ecg_base_folder, subject_folder)

            if not os.path.isdir(video_folder_path) or not os.path.isdir(ecg_folder_path):
                continue

            video_folders = sorted(os.listdir(video_folder_path))
            ecg_files = sorted(os.listdir(ecg_folder_path))

            for video_idx in range(len(video_folders)):
                video_path = os.path.join(video_folder_path, video_folders[video_idx])
                ecg_file = ecg_files[video_idx] if video_idx < len(ecg_files) else None

                # Check if the video folder has frames and the ECG file exists
                if ecg_file is None or not os.path.isdir(video_path):
                    continue

                frame_files = sorted([f for f in os.listdir(video_path) if f.endswith('.jpg') or f.endswith('.png')])
                if len(frame_files) == 0:
                    # Skip if there are no frames
                    continue

                valid_data.append((video_path, os.path.join(ecg_folder_path, ecg_file)))

        return valid_data

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        video_path, ecg_path = self.valid_data[idx]

        # Load video frames
        frame_files = sorted([os.path.join(video_path, f) for f in os.listdir(video_path) if f.endswith('.jpg') or f.endswith('.png')])

        # Sample or pad frames to get exactly max_frames (138 frames)
        if len(frame_files) > self.max_frames:
            indices = np.linspace(0, len(frame_files) - 1, self.max_frames, dtype=int)
            frame_files = [frame_files[i] for i in indices]
        else:
            frame_files += [frame_files[0]] * (self.max_frames - len(frame_files))

        frames = torch.stack([transform(Image.open(frame).convert("RGB")) for frame in frame_files])

        # Load ECG data
        ecg_data = pd.read_csv(ecg_path, header=None, skiprows=2).iloc[:, 1].values

        # Pad or truncate ECG data to max_length
        if len(ecg_data) > self.max_length:
            ecg_data = ecg_data[:self.max_length]
        else:
            ecg_data = np.pad(ecg_data, (0, self.max_length - len(ecg_data)), 'constant')

        ecg_tensor = torch.tensor(ecg_data, dtype=torch.float32).unsqueeze(-1)

        # Determine the label based on the ECG file name
        if 'BL' in ecg_path:
            label = 0
        elif 'PA1' in ecg_path:
            label = 1
        elif 'PA2' in ecg_path:
            label = 2
        elif 'PA3' in ecg_path:
            label = 3
        else:
            label = 0

        return frames, ecg_tensor, label

# Load the models
facial_model = FacialVideoEncoder().to(device)
facial_model.load_state_dict(torch.load("facial_video_encoder.pth"))
facial_model.eval()

ecg_model = HeartRateEncoder().to(device)
ecg_model.load_state_dict(torch.load("heart_rate_encoder_model.pth"))
ecg_model.eval()

# Function to get combined predictions
def get_combined_predictions(video_input, ecg_input, w_A=0.8, w_B=0.2):
    with torch.no_grad():
        logits_A = facial_model(video_input)
        logits_B = ecg_model(ecg_input)
        combined_logits = w_A * logits_A + w_B * logits_B
        combined_probabilities = F.softmax(combined_logits, dim=1)
        _, predicted_class = torch.max(combined_probabilities, dim=1)
    return predicted_class

# Calculate combined accuracy
def calculate_combined_accuracy(data_loader):
    total_samples = 0
    correct_predictions = 0

    for video_inputs, ecg_inputs, labels in data_loader:
        video_inputs, ecg_inputs, labels = video_inputs.to(device), ecg_inputs.to(device), labels.to(device)
        predicted_class = get_combined_predictions(video_inputs, ecg_inputs)
        total_samples += labels.size(0)
        correct_predictions += (predicted_class == labels).sum().item()

    accuracy = 100.0 * correct_predictions / total_samples
    return accuracy

# Main function
if __name__ == '__main__':
    video_base_folder = r"prepro_vid1"
    ecg_base_folder = r"ecg_only1"
    dataset = CombinedBioVidDataset(video_base_folder, ecg_base_folder, max_frames=138)
    val_loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    accuracy = calculate_combined_accuracy(val_loader)
    print(f"Combined Model Accuracy: {accuracy:.2f}%")
