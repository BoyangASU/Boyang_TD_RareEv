import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from typing import Optional

class RULDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class CNN1D(nn.Module):
    def __init__(self, input_channels: int):
        super(CNN1D, self).__init__()
        
        # Device selection logic
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
            
        print(f"Using device: {self.device}")
        
        self.conv1 = nn.Conv1d(input_channels, 256, kernel_size=7)
        self.conv2 = nn.Conv1d(256, 96, kernel_size=7)
        self.conv3 = nn.Conv1d(96, 32, kernel_size=7)
        
        self.relu = nn.ReLU()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 1)
        
        self.to(self.device)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x.squeeze()

def train_model(train_data: np.ndarray, 
                train_targets: np.ndarray,
                batch_size: int = 64,
                epochs: int = 20,
                learning_rate: float = 0.001,
                model_save_path: str = "saved_models",
                model_name: str = "model") -> CNN1D:
    
    # Create save directory if it doesn't exist
    os.makedirs(model_save_path, exist_ok=True)
    
    # Split data into train and validation sets
    train_data, val_data, train_targets, val_targets = train_test_split(
        train_data, train_targets, test_size=0.2, random_state=83
    )
    
    # Create datasets and dataloaders
    train_dataset = RULDataset(train_data, train_targets)
    val_dataset = RULDataset(val_data, val_targets)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model and optimizer
    model = CNN1D(input_channels=train_data.shape[2])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Training loop
    best_val_loss = float('inf')
    print(f"Training {model_name} on {model.device}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{epochs}] Training')
        
        for batch_features, batch_targets in train_pbar:
            batch_features = batch_features.to(model.device)
            batch_targets = batch_targets.to(model.device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features = batch_features.to(model.device)
                batch_targets = batch_targets.to(model.device)
                
                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(model_save_path, f'best_{model_name}.pth'))
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(model_save_path, f'best_{model_name}.pth')))
    return model

def plot_predictions(true_rul: np.ndarray, pred_rul: np.ndarray, save_path: str = None):
    """Plot true and predicted RUL values."""
    plt.figure(figsize=(10, 6))
    
    # Sort by true RUL
    sort_idx = np.argsort(true_rul)
    true_rul_sorted = true_rul[sort_idx]
    pred_rul_sorted = pred_rul[sort_idx]
    
    plt.plot(true_rul_sorted, label="True RUL", color="red")
    plt.plot(pred_rul_sorted, label="Predicted RUL", color="blue")
    plt.xlabel("Sorted Engine Index")
    plt.ylabel("RUL")
    plt.title("True vs Predicted RUL (Sorted by True RUL)")
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def compute_s_score(rul_true: np.ndarray, rul_pred: np.ndarray) -> float:
    """Compute the S-score metric."""
    diff = rul_pred - rul_true
    return np.sum(np.where(diff < 0, np.exp(-diff/13)-1, np.exp(diff/10)-1))

def train_td_epoch(model: CNN1D,
                  train_loader: DataLoader,
                  device: torch.device,
                  optimizer: torch.optim.Optimizer,
                  gamma: float = 0.999,
                  use_piecewise: bool = True,
                  early_rul: Optional[int] = 125,
                  window_length: int = 30) -> float:
    """Train one epoch using TD(0) learning"""
    model.train()
    total_loss = 0
    num_batches = 0
    criterion = nn.MSELoss()
    
    for batch_idx, (features, targets) in enumerate(train_loader):
        features = features.to(device)
        targets = targets.to(device)
        batch_size = features.size(0)
        sequence_length = features.size(1)
        
        if len(targets.shape) == 1:
            targets = targets.view(-1, 1).repeat(1, sequence_length)

        for t in range(sequence_length - window_length):
            current_state = features[:, t:t+window_length, :]
            next_state = features[:, t+1:t+1+window_length, :]
            
            current_rul = targets[:, t]
            next_rul = targets[:, t+1]
            
            reward = -(current_rul - next_rul)
            
            if use_piecewise and early_rul is not None:
                current_rul = torch.where(current_rul > early_rul, 
                                        torch.tensor(early_rul, device=device), 
                                        current_rul)
                next_rul = torch.where(next_rul > early_rul,
                                      torch.tensor(early_rul, device=device),
                                      next_rul)
            
            current_value = model(current_state).squeeze()
            
            with torch.no_grad():
                next_value = model(next_state).squeeze()
                td_target = reward + gamma * next_value
            
            # Debugging Information
            print(f"\nBatch {batch_idx}, Step {t}")
            print(f"Current RUL: {current_rul}")
            print(f"Next RUL: {next_rul}")
            print(f"Reward: {reward}")
            print(f"Current Value: {current_value}")
            print(f"Next Value: {next_value}")
            print(f"TD Target: {td_target}")
            print(f"Loss before backward: {criterion(current_value, td_target).item():.4f}")
            
            loss = criterion(current_value, td_target)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0

def train_model_td(model: CNN1D,
                  train_loader: DataLoader,
                  val_loader: Optional[DataLoader] = None,
                  epochs: int = 20,
                  learning_rate: float = 0.0001,
                  gamma: float = 0.999,
                  use_piecewise: bool = True,
                  early_rul: Optional[int] = 125,
                  window_length: int = 30,
                  model_save_path: str = "saved_models",
                  model_name: str = "td_model") -> CNN1D:
    """Train model using TD(0) learning"""
    device = model.device
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = float('inf')
    
    print(f"Training TD model with {'piecewise' if use_piecewise else 'linear'} RUL")
    print(f"Device: {device}, Gamma: {gamma}")
    
    for epoch in range(epochs):
        loss = train_td_epoch(
            model=model,
            train_loader=train_loader,
            device=device,
            optimizer=optimizer,
            gamma=gamma,
            use_piecewise=use_piecewise,
            early_rul=early_rul,
            window_length=window_length
        )
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss:.4f}')
        
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), 
                      os.path.join(model_save_path, f'best_{model_name}.pth'))
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(model_save_path, f'best_{model_name}.pth')))
    return model
