#%%
import pdb
import torch
import numpy as np
from torch.utils.data import DataLoader
from train import CNN1D, RULDataset
from preprocessing import prepare_data, process_input_data_with_targets, process_targets
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import mean_squared_error
from oversample_terminal import create_terminal_oversampled_loader  # New import for oversampling

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
        targets  = targets.unsqueeze(0).to(model.device)
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

def train_td_model(train_data: np.ndarray, 
                   train_targets: np.ndarray,
                   batch_size: int = 32,
                   model_name: str = "td_model",
                   use_piecewise: bool = True,
                   early_rul: int = 125,
                   window_size: int = 30,
                   use_oversampling: bool = True,
                   gamma: float = 0.99
                   ) -> CNN1D:
    """
    Train a TD model (either Piecewise or Linear) using a conventional TD-learning update.
    The training now computes the TD error over all sliding windows in each training sequence.

    Parameters:
      train_data: Preprocessed training data with shape (num_samples, window_length, num_features).
      train_targets: Preprocessed RUL targets.
      batch_size: Training batch size.
      model_name: Used to save the model checkpoint.
      use_piecewise: If True, use piecewise reward functions.
      early_rul: Threshold for RUL in target processing.
      window_size: Length of the sliding window.
      use_oversampling: If True, uses oversampling for terminal transitions; otherwise, uses standard sampling.
      gamma: Discount factor for the TD update.
    
    Returns:
      model: Trained CNN1D model.
    """
    os.makedirs("saved_models", exist_ok=True)
    
    # Split into training and validation sets.
    X_train, X_val, y_train, y_val = train_test_split(train_data, train_targets, test_size=0.2, random_state=83)
    
    train_dataset = RULDataset(X_train, y_train)
    val_dataset   = RULDataset(X_val, y_val)
    
    # Use oversampling or a standard DataLoader based on the flag.
    if use_oversampling:
        train_loader = create_terminal_oversampled_loader(
            train_dataset, 
            terminal_threshold=10.0, 
            oversample_multiplier=5.0, 
            batch_size=batch_size
        )
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
    model = CNN1D(input_channels=train_data.shape[2])
    # Update the device selection to check for MPS if available.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    epochs = 20
    best_val_error = np.inf
    
    for epoch in range(epochs):
        model.train()
        total_td_loss = 0.0
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            
            # Ensure batch_targets is at least 2D.
            if batch_targets.dim() == 1:
                batch_targets = batch_targets.unsqueeze(1)
            
            sequence_length = batch_features.size(1)
            num_windows = sequence_length - window_size + 1
            
            # Iterate over each sliding window within the sequence.
            for t in range(num_windows):
                current_state = batch_features[:, t:t+window_size, :]
                # The target corresponds to the last element of the current window.
                current_rul = batch_targets[:, t+window_size-1]
                
                current_value = model(current_state).squeeze()
                td_target = torch.empty_like(current_rul, device=device)
                
                # Determine terminal and non-terminal transitions.
                terminal_mask = (current_rul == 0)
                non_terminal_mask = ~terminal_mask
                td_target[terminal_mask] = 0.0
                next_state = batch_features[non_terminal_mask, t+1:t+1+window_size, :]
                next_value = model(next_state).detach().squeeze()
                if use_piecewise:
                    reward_non_terminal = torch.where(
                        current_rul[non_terminal_mask] > early_rul,
                        torch.tensor(0.0, device=device),
                        torch.tensor(1.0, device=device))
                else:
                    reward_non_terminal = torch.ones_like(current_rul[non_terminal_mask], device=device)
                td_target[non_terminal_mask] = reward_non_terminal + gamma * next_value
                
                loss = criterion(current_value, td_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_td_loss += loss.item()
                
        avg_td_loss = total_td_loss / len(train_dataset)
        true_val_error = compute_true_val_error(model, val_dataset, window_size)
        print(f"Epoch [{epoch+1}/{epochs}] TD Loss: {avg_td_loss:.4f}, Val RMSE: {true_val_error:.4f}")
        
        # Save the best model based on validation error.
        if true_val_error < best_val_error:
            best_val_error = true_val_error
            torch.save(model.state_dict(), f"saved_models/best_{model_name}.pth")
            print("Best Model Saved")
    
    model.load_state_dict(torch.load(f"saved_models/best_{model_name}.pth"))
    return model

def main():
    """
    Main execution for TD algorithm.
    Uses the updated preprocessing functions.
    """
    # Prepare data with explicit parameters.
    train_data, test_data, true_rul, sequence_length, shift, early_rul = prepare_data(
        train_path="../CMAPSSData/train_FD001.txt",
        test_path="../CMAPSSData/test_FD001.txt",
        rul_path="../CMAPSSData/RUL_FD001.txt",
        window_length=31,
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
    
    # For TD Piecewise:
    td_piecewise_model = train_td_model(
        processed_train_data, 
        processed_train_targets, 
        model_name="td_piecewise",  # This ensures the file is saved as best_td_piecewise.pth
        use_piecewise=True, 
        early_rul=early_rul,
        window_size=30,
        use_oversampling=False  # Add this parameter
    )
    
    # For TD Linear, you would call:
    td_linear_model = train_td_model(
        processed_train_data, 
        processed_train_targets, 
        model_name="td_linear",  # Corresponds to best_td_linear.pth
        use_piecewise=False, 
        early_rul=early_rul,
        window_size=30,
        use_oversampling=False  # Add this parameter
    )

if __name__ == "__main__":
    print("Running TD algorithm main()")
    main() 