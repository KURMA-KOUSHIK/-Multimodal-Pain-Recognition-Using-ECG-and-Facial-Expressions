import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt
from torch.utils.data import Dataset, DataLoader, random_split

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### 1. ECG Preprocessing and Feature Extraction ###

def preprocess_ecg_signal(ecg_signal, fs=512):
    """Preprocess ECG signal with bandpass filtering and QRS complex detection."""
    # Bandpass filter parameters
    lowcut = 0.5
    highcut = 40.0
    
    # Design bandpass filter
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(1, [low, high], btype="band")
    
    # Apply the bandpass filter
    filtered_ecg = filtfilt(b, a, ecg_signal)
    
    # R-peak detection
    peaks, _ = find_peaks(filtered_ecg, distance=fs//2)
    
    # Calculate heart rate from R-R intervals
    rr_intervals = np.diff(peaks) / fs * 1000  # Convert to ms
    heart_rate = 60000 / rr_intervals  # Heart rate in BPM
    
    return heart_rate

### 2. Custom Dataset for Multilevel Pain Classification ###

class BioVidECGDataset(Dataset):
    def __init__(self, base_folder, max_length=100):
        super(BioVidECGDataset, self).__init__()
        self.base_folder = base_folder
        self.ecg_files = []
        self.labels = []
        self.max_length = max_length  # Fixed length for all tensors
        self._load_files()

    def _load_files(self):
        for subject_folder in os.listdir(self.base_folder):
            subject_path = os.path.join(self.base_folder, subject_folder)
            if os.path.isdir(subject_path):
                for file_name in os.listdir(subject_path):
                    if file_name.endswith('.csv'):
                        file_path = os.path.join(subject_path, file_name)
                        label = self._get_label_from_filename(file_name)
                        self.ecg_files.append(file_path)
                        self.labels.append(label)

    def _get_label_from_filename(self, filename):
        """Return pain level based on filename."""
        if 'BL1' in filename:
            return 0  # No pain
        elif 'PA1' in filename:
            return 1  # Mild pain
        elif 'PA2' in filename:
            return 2  # Moderate pain
        elif 'PA3' in filename:
            return 3  # Severe pain
        else:
            return 0  # Default to no pain

    def __len__(self):
        return len(self.ecg_files)

    def __getitem__(self, idx):
        ecg_path = self.ecg_files[idx]
        label = self.labels[idx]
        
        # Read ECG data from the 3rd row onward (skip first 2 rows)
        ecg_data = pd.read_csv(ecg_path, header=None, skiprows=2).iloc[:, 1].values
        
        # Preprocess the ECG data to extract heart rate
        heart_rate = preprocess_ecg_signal(ecg_data)
        
        # Pad or truncate heart rate to the fixed length
        if len(heart_rate) > self.max_length:
            heart_rate = heart_rate[:self.max_length]
        else:
            heart_rate = np.pad(heart_rate, (0, self.max_length - len(heart_rate)), 'constant')
        
        # Convert heart rate to tensor
        heart_rate_tensor = torch.tensor(heart_rate, dtype=torch.float32).unsqueeze(-1)
        
        return heart_rate_tensor, torch.tensor(label, dtype=torch.long)

### 3. Heart Rate Encoder Model for Multiclass Output ###

class HeartRateEncoder(nn.Module):
    def __init__(self, input_dim=1, embed_dim=512, num_heads=8, num_layers=2, dropout=0.3, num_classes=4):
        super(HeartRateEncoder, self).__init__()
        self.input_projection = nn.Linear(input_dim, embed_dim)
        
        # Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Output layer for multiclass classification
        self.output_layer = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x

### 4. Training and Validation Functions ###

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, save_path="heart_rate_encoder_model.pth"):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0

        for signals, labels in train_loader:
            signals, labels = signals.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(signals)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            epoch_loss += loss.item()

        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Accuracy: {accuracy:.2f}%")

        # Validation phase
        val_loss, val_accuracy = validate_model(model, val_loader, criterion)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    # Save the trained model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def validate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for signals, labels in val_loader:
            signals, labels = signals.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(signals)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            # Accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total
    return val_loss, val_accuracy

### 5. Main Execution with Train/Validation Split ###

def main(base_folder):
    # Create dataset and split into training and validation sets
    dataset = BioVidECGDataset(base_folder, max_length=2800)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = HeartRateEncoder(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer)

# Execute the main function with the base folder path
if __name__ == '__main__':
    base_folder = r"F:\sem5\Biomedical signal Processing\project_related\part_A\ecg_only"
    main(base_folder)
    model = HeartRateEncoder(input_dim=1, embed_dim=512, num_heads=8, num_layers=2, dropout=0.3, num_classes=4)

# Count total trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

