import torch
import numpy as np
from torch.utils.data import DataLoader
from train import CNN1D, RULDataset
from preprocessing import prepare_data, process_input_data_with_targets, process_targets
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

def compute_true_val_error(model, val_dataset, window_size):
    """
    Compute the 'true' validation error by forwarding the entire validation dataset,
    extracting the prediction from the last sliding window, and comparing it to the true RUL.
    
    Parameters:
      model: The trained model.
      val_dataset: The validation dataset (an instance of RULDataset or similar).
      window_size: The size of the sliding window.
    
    Returns:
      rmse: The RMSE computed over the validation engines.
    """
    model.eval()
    all_preds = []
    all_true = []
    
    # The following loop assumes that each sample in your dataset corresponds to one engine's full sequence.
    for features, targets in val_dataset:
        features = features.unsqueeze(0).to(model.device)  # Add a batch dimension
        targets = targets.unsqueeze(0).to(model.device)
        sequence_length = features.size(1)
        num_windows = sequence_length - window_size + 1
        
        # Extract prediction from the last sliding window.
        last_window = features[:, num_windows - 1:num_windows - 1 + window_size, :]
        pred = model(last_window).squeeze().detach().cpu().item()
        true_val = targets[:, -1].squeeze().detach().cpu().item()  # Assuming last value is the true RUL
        
        all_preds.append(pred)
        all_true.append(true_val)
    
    rmse = np.sqrt(mean_squared_error(all_true, all_preds))
    return rmse

class FeatureExtractorCNN(torch.nn.Module):
    """Feature extractor that uses CNN layers but outputs features instead of values"""
    def __init__(self, input_channels):
        super(FeatureExtractorCNN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cnn_layers = torch.nn.Sequential(
            torch.nn.Conv1d(input_channels, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(16, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
        ).to(self.device)
        
    def forward(self, x):
        # Swap dimensions for CNN: [batch, seq_len, features] -> [batch, features, seq_len]
        x = x.transpose(1, 2).to(self.device)
        features = self.cnn_layers(x)
        # Flatten the output for the linear layer
        return features.mean(dim=2)  # Global average pooling

class LinearValueFunction(torch.nn.Module):
    """Linear value function that takes features and outputs a value"""
    def __init__(self, input_size=64):
        super(LinearValueFunction, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.linear = torch.nn.Linear(input_size, 1).to(self.device)
        
    def forward(self, x):
        return self.linear(x).squeeze()

def train_linear_td_lambda_model(train_data: np.ndarray, 
                                 train_targets: np.ndarray,
                                 batch_size: int = 1,
                                 epochs: int = 20,
                                 learning_rate: float = 0.001,
                                 model_save_path: str = "saved_models",
                                 model_name: str = "linear_td_lambda",
                                 use_piecewise: bool = True,
                                 early_rul: int = 125,
                                 gamma: float = 0.999,
                                 lambd: float = 0.9,
                                 window_size: int = 30) -> tuple:
    """
    Train using TD(λ) only on the last linear layer for efficiency.
    """
    os.makedirs(model_save_path, exist_ok=True)
    
    # Split data into training and validation sets
    train_data, val_data, train_targets, val_targets = train_test_split(
        train_data, train_targets, test_size=0.2, random_state=83
    )
    
    # Create datasets and dataloaders
    train_dataset = RULDataset(train_data, train_targets)
    val_dataset = RULDataset(val_data, val_targets)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize models - separate feature extractor from value function
    feature_extractor = FeatureExtractorCNN(input_channels=train_data.shape[2])
    value_function = LinearValueFunction(input_size=64)  # Match feature size
    
    # Freeze the feature extractor parameters
    for param in feature_extractor.parameters():
        param.requires_grad = False
    
    best_val_error = float('inf')
    
    print("Starting Linear TD(λ) training...")
    for epoch in range(epochs):
        total_loss = 0.0
        
        # Pre-compute features for validation set to speed up evaluation
        val_features = []
        val_targets_list = []
        with torch.no_grad():
            feature_extractor.eval()
            for features, targets in val_dataset:
                features = features.unsqueeze(0)
                window_features = []
                sequence_length = features.size(1)
                num_windows = sequence_length - window_size + 1
                
                for t in range(num_windows):
                    current_state = features[:, t:t+window_size, :]
                    window_features.append(feature_extractor(current_state))
                
                val_features.append(torch.stack(window_features, dim=0).squeeze(1))
                val_targets_list.append(targets[-1].item())  # Last RUL value
        
        # Training loop
        feature_extractor.eval()  # Keep in eval mode since we're not training it
        value_function.train()
        
        for batch_features, batch_targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch_features = batch_features.to(feature_extractor.device)
            batch_targets = batch_targets.to(value_function.device)
            
            sequence_length = batch_features.size(1)
            num_windows = sequence_length - window_size + 1
            if num_windows < 1:
                continue
            
            # Pre-compute all features for the episode to avoid redundant forward passes
            precomputed_features = []
            with torch.no_grad():
                for t in range(num_windows):
                    current_state = batch_features[:, t:t+window_size, :]
                    features = feature_extractor(current_state)
                    precomputed_features.append(features)
            
            # Initialize eligibility traces (only for value function parameters)
            eligibility_traces = {name: torch.zeros_like(param, device=value_function.device)
                               for name, param in value_function.named_parameters()}
            
            # Get initial value
            v_old = value_function(precomputed_features[0]).squeeze()
            
            episode_loss = 0.0
            for t in range(num_windows):
                # Use precomputed features
                current_features = precomputed_features[t]
                v = value_function(current_features).squeeze()
                
                # Compute reward
                current_rul = batch_targets[:, t+window_size-1]
                if use_piecewise:
                    reward = torch.where(current_rul > early_rul,
                                      torch.tensor(0.0, device=value_function.device),
                                      torch.tensor(1.0, device=value_function.device))
                else:
                    reward = torch.ones_like(current_rul, device=value_function.device)
                
                # Compute next state value
                if t < num_windows - 1:
                    next_features = precomputed_features[t+1]
                    v_next = value_function(next_features).squeeze()
                else:
                    v_next = torch.tensor(0.0, device=value_function.device)
                
                # TD error
                delta = reward + gamma * v_next - v
                
                # Compute gradient of value function only
                value_function.zero_grad()
                v.backward(retain_graph=True)
                
                # Collect gradients
                grads = {}
                for name, param in value_function.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        grads[name] = param.grad.detach().clone()
                
                # Update eligibility traces (only for value function)
                for name, param in value_function.named_parameters():
                    if param.requires_grad:
                        e = eligibility_traces[name]
                        grad = grads[name]
                        dot_prod = torch.sum(e.view(-1) * grad.view(-1))
                        eligibility_traces[name] = gamma * lambd * e + (1 - learning_rate * gamma * lambd * dot_prod) * grad
                
                # Update value function weights
                for name, param in value_function.named_parameters():
                    if param.requires_grad:
                        grad = grads[name]
                        e = eligibility_traces[name]
                        update = learning_rate * (delta + (v - v_old)) * e - learning_rate * (v - v_old) * grad
                        param.data.add_(update)
                
                v_old = v_next.detach()
                episode_loss += (delta.item())**2
            
            total_loss += episode_loss
        
        avg_train_loss = total_loss / len(train_loader.dataset)
        
        # Evaluate on validation set
        value_function.eval()
        val_preds = []
        with torch.no_grad():
            for features in val_features:
                # Use the last window's features for prediction
                val_pred = value_function(features[-1]).item()
                val_preds.append(val_pred)
        
        true_val_error = np.sqrt(mean_squared_error(val_targets_list, val_preds))
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val RUL Error (RMSE): {true_val_error:.4f}")
        
        if true_val_error < best_val_error:
            best_val_error = true_val_error
            torch.save({
                'feature_extractor': feature_extractor.state_dict(),
                'value_function': value_function.state_dict()
            }, os.path.join(model_save_path, f"best_{model_name}.pth"))
            print("Best Model Saved")
    
    # Load best model
    checkpoint = torch.load(os.path.join(model_save_path, f"best_{model_name}.pth"))                                                                                                                                                                                                                                                           
    feature_extractor.load_state_dict(checkpoint['feature_extractor'])
    value_function.load_state_dict(checkpoint['value_function'])
    
    return feature_extractor, value_function

def main():
    """
    Main execution for TD algorithm variants.
    This main function demonstrates data preparation, training both the traditional TD and
    True Online TD(λ) models (if desired), and saving the trained models.
    """
    # Prepare data with explicit parameters.
    train_data, test_data, true_rul, sequence_length, shift, early_rul = prepare_data(
        train_path="../CMAPSSData/train_FD001.txt",
        test_path="../CMAPSSData/test_FD001.txt",
        rul_path="../CMAPSSData/RUL_FD001.txt",
        window_length=200,
        shift=1,
        early_rul=125
    )
    
    processed_train_data, processed_train_targets = [], []
    num_train_machines = len(train_data[0].unique())
    
    for i in range(1, num_train_machines + 1):
        # Get sensor-only data for engine i.
        temp_train_data = train_data[train_data[0] == i].drop(columns=[0]).values
        temp_train_targets = process_targets(data_length=temp_train_data.shape[0], early_rul=early_rul)
        
        # Create sliding windows using the full sequence length.
        data_for_machine, targets_for_machine = process_input_data_with_targets(
            temp_train_data, temp_train_targets, window_length=sequence_length, shift=shift, target_for_all_timestamps=True
        )
        processed_train_data.append(data_for_machine)
        processed_train_targets.append(targets_for_machine)
    
    processed_train_data = np.concatenate(processed_train_data)
    processed_train_targets = np.concatenate(processed_train_targets)
    
    print(f"Processed Train Data Shape: {processed_train_data.shape}")
    print(f"Processed Train Targets Shape: {processed_train_targets.shape}")
    
    # Train the model using Linear TD(λ)
    feature_extractor, value_function = train_linear_td_lambda_model(
        processed_train_data, 
        processed_train_targets, 
        batch_size=1,
        model_name="linear_td_lambda",
        use_piecewise=True,
        early_rul=early_rul,
        window_size=50,  # Increased window size to capture more context
        gamma=0.99,
        lambd=0,
        learning_rate=0.01,
        epochs=20
    )
    
if __name__ == "__main__":
    print("Running TD algorithm main() with Linear TD(λ)")
    main() 

# for solar, reverse the reward
# average reward for continuous