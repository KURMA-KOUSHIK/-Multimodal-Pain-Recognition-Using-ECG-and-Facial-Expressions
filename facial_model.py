
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models import resnet18
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### 1. Custom Dataset for Facial Video Frames ###

class BioVidFacialDataset(Dataset):
    def __init__(self, base_folder, transform=None, max_frames=16):
        super(BioVidFacialDataset, self).__init__()
        self.base_folder = base_folder
        self.video_folders = []
        self.labels = []
        self.max_frames = max_frames
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self._load_files()

    def _load_files(self):
        """Load video folders and labels."""
        for subject_folder in os.listdir(self.base_folder):
            subject_path = os.path.join(self.base_folder, subject_folder)
            if os.path.isdir(subject_path):
                for video_folder in os.listdir(subject_path):
                    video_path = os.path.join(subject_path, video_folder)
                    if os.path.isdir(video_path):
                        frame_files = self._get_frame_files(video_path)
                        if frame_files:  # Only add if frames are found
                            self.video_folders.append(video_path)
                            label = self._get_label_from_folder(video_folder)
                            self.labels.append(label)

    def _get_label_from_folder(self, folder_name):
        """Map folder names to labels."""
        if 'P0' in folder_name:
            return 0
        elif 'P1' in folder_name:
            return 1
        elif 'P2' in folder_name:
            return 2
        elif 'P3' in folder_name:
            return 3
        else:
            return 0

    def _get_frame_files(self, video_path):
        """Retrieve and sort frame files."""
        frame_files = sorted(
            [os.path.join(video_path, f) for f in os.listdir(video_path)
             if f.endswith('.jpg') or f.endswith('.png')]
        )
        return frame_files

    def __len__(self):
        return len(self.video_folders)

    def __getitem__(self, idx):
        video_path = self.video_folders[idx]
        label = self.labels[idx]

        # Get sorted frame files
        frame_files = self._get_frame_files(video_path)
        if len(frame_files) == 0:
            raise ValueError(f"No frames found in {video_path}")

        # Handle cases where frames are fewer than max_frames
        if len(frame_files) > self.max_frames:
            frame_files = frame_files[:self.max_frames]
        else:
            frame_files += [frame_files[-1]] * (self.max_frames - len(frame_files))

        # Load frames and apply transforms
        frames = torch.stack([self.transform(Image.open(frame).convert("RGB")) for frame in frame_files])
        return frames, torch.tensor(label, dtype=torch.long)

### 2. Model for Video-Based Pain Classification ###

class FacialVideoEncoder(nn.Module):
    def __init__(self, num_classes=4, max_frames=16):
        super(FacialVideoEncoder, self).__init__()
        self.max_frames = max_frames
        self.feature_extractor = resnet18(pretrained=True)
        self.feature_extractor.fc = nn.Identity()
        self.temporal_encoder = nn.GRU(input_size=512, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True)
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

### 3. Training and Validation Functions ###

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for videos, labels in train_loader:
            videos, labels = videos.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_accuracy = 100.0 * correct / total
        val_loss, val_accuracy = validate_model(model, val_loader, criterion)
        scheduler.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    save_model(model, "facial_video_encoder.pth")

def validate_model(model, val_loader, criterion):
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for videos, labels in val_loader:
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss /= len(val_loader)
    val_accuracy = 100.0 * correct / total
    return val_loss, val_accuracy

def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")

### 5. Main Function ###

def main(base_folder):
    dataset = BioVidFacialDataset(base_folder)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

    model = FacialVideoEncoder(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler)

# Entry point
if __name__ == '__main__':
    base_folder = r"prepro_vid"
    main(base_folder)
