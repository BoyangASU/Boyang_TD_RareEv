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
    Compute the 'true' validation error by passing the entire validation dataset
    through the model and extracting the prediction from the last sliding window
    for each engine.

    Args:
        model (torch.nn.Module): Trained model used to make predictions.
        val_dataset (torch.utils.data.Dataset): Validation dataset 
            (e.g., an RULDataset instance).
        window_size (int): The length of each sliding window to feed into the model.

    Returns:
        float: The RMSE computed across all engines (samples) in val_dataset.
    """
    model.eval()
    all_preds = []
    all_true = []
    
    # Each sample in val_dataset is assumed to correspond to one engine's full data sequence.
    for features, targets in val_dataset:
        # Insert a batch dimension for model inference: (1, seq_len, feature_dim)
        features = features.unsqueeze(0).to(model.device)
        targets = targets.unsqueeze(0).to(model.device)
        
        sequence_length = features.size(1)
        num_windows = sequence_length - window_size + 1
        
        # Extract prediction from the last sliding window
        last_window = features[:, num_windows - 1 : num_windows - 1 + window_size, :]
        pred = model(last_window).squeeze().detach().cpu().item()
        
        # We assume the last target in 'targets' is the ground truth RUL for this sequence
        true_val = targets[:, -1].squeeze().detach().cpu().item()
        
        all_preds.append(pred)
        all_true.append(true_val)
    
    rmse = np.sqrt(mean_squared_error(all_true, all_preds))
    return rmse


class FeatureExtractorCNN(torch.nn.Module):
    """
    Feature extractor using convolutional layers, returning a feature vector
    rather than a direct RUL or value.

    Args:
        input_channels (int): Number of input features (sensor channels).
    """
    def __init__(self, input_channels):
        super(FeatureExtractorCNN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Define the CNN layers
        self.cnn_layers = torch.nn.Sequential(
            torch.nn.Conv1d(input_channels, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(16, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
        ).to(self.device)
        
    def forward(self, x):
        """
        Forward pass through the feature extractor.

        Args:
            x (torch.Tensor): Shape (batch_size, seq_length, feature_dim).

        Returns:
            torch.Tensor: A feature tensor of shape (batch_size, 64).
        """
        # Rearrange input to (batch_size, feature_dim, seq_length) for Conv1D
        x = x.transpose(1, 2).to(self.device)
        features = self.cnn_layers(x)
        # Perform global average pooling over the temporal dimension
        return features.mean(dim=2)


class LinearValueFunction(torch.nn.Module):
    """
    A simple linear layer that takes a feature vector and outputs a scalar value
    (e.g., the predicted RUL or a value function in TD learning).

    Args:
        input_size (int): Dimension of incoming feature vectors.
    """
    def __init__(self, input_size=64):
        super(LinearValueFunction, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.linear = torch.nn.Linear(input_size, 1).to(self.device)
        
    def forward(self, x):
        """
        Forward pass for the linear head.

        Args:
            x (torch.Tensor): Feature tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Scalar predictions (batch_size,).
        """
        return self.linear(x).squeeze()


def train_linear_td_lambda_model(
    train_data: np.ndarray, 
    train_targets: np.ndarray,
    batch_size: int = 32,
    epochs: int = 20,
    learning_rate: float = 0.001,
    model_save_path: str = "saved_models",
    model_name: str = "linear_td_lambda",
    use_piecewise: bool = True,
    early_rul: int = 125,
    gamma: float = 0.999,
    lambd: float = 0.9,
    window_size: int = 30
) -> tuple:
    """
    Train a CNN-based feature extractor + linear head with TD(λ) updates, focusing on
    updating only the linear head parameters. This version demonstrates usage of the 
    torch.func.vmap and torch.func.grad APIs for per-sample gradient computation.

    Args:
        train_data (np.ndarray): Training input of shape (num_samples, seq_length, num_features).
        train_targets (np.ndarray): Corresponding RUL/target values. Shape (num_samples, seq_length).
        batch_size (int): Number of sequences per training batch.
        epochs (int): Number of training epochs.
        learning_rate (float): Step size for parameter updates.
        model_save_path (str): Directory where the best model checkpoint will be saved.
        model_name (str): Identifier for the saved checkpoint file.
        use_piecewise (bool): Whether to apply a piecewise reward (0 if RUL>early_rul, else 1).
        early_rul (int): Threshold for the piecewise reward.
        gamma (float): Discount factor for TD updates.
        lambd (float): λ parameter for eligibility traces.
        window_size (int): The size of each sliding window for computing RUL.

    Returns:
        (FeatureExtractorCNN, LinearValueFunction):
            The trained feature extractor and value function (with best validation RMSE).
    """
    # For better code clarity, check if PyTorch 2.0 or higher is available.
    try:
        from torch.func import vmap, grad, functional_call
    except ImportError:
        raise ImportError("torch.func APIs are required. Please use PyTorch 2.0 or newer.")

    # Split data into train/val sets
    train_data, val_data, train_targets, val_targets = train_test_split(
        train_data, train_targets, test_size=0.2, random_state=83
    )
    
    # Create datasets and dataloaders
    train_dataset = RULDataset(train_data, train_targets)
    val_dataset   = RULDataset(val_data, val_targets)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize the feature extractor and the linear head
    feature_extractor = FeatureExtractorCNN(input_channels=train_data.shape[2])
    value_function = LinearValueFunction(input_size=64)
    
    # Freeze all parameters in the feature extractor to only train the linear layer
    for param in feature_extractor.parameters():
        param.requires_grad = False

    # Convert value function parameters into a tuple for functional API usage
    params = tuple(value_function.parameters())
    
    # -- Define helpers for functional forward passes --
    def f_value_function(params, features):
        """
        Functional forward pass through the value function, bypassing 'in-place' usage of .parameters().
        
        Args:
            params (tuple): Current parameter values for the linear layer.
            features (torch.Tensor): Input features of shape (batch_size, feature_dim).

        Returns:
            torch.Tensor: Output value, shape (batch_size,).
        """
        param_dict = {
            name: p for (name, _), p in zip(value_function.named_parameters(), params)
        }
        return functional_call(value_function, param_dict, features)
    
    def single_sample_value(params, sample):
        """
        Compute a scalar value for a single sample. This function helps with vmap-based per-sample gradients.

        Args:
            params (tuple): Current parameter values for the linear layer.
            sample (torch.Tensor): Feature vector of shape (feature_dim,).

        Returns:
            torch.Tensor: Scalar value for one sample.
        """
        param_dict = {
            name: p for (name, _), p in zip(value_function.named_parameters(), params)
        }
        # We unsqueeze to mimic (batch=1, feature_dim), then squeeze back.
        return functional_call(value_function, param_dict, sample.unsqueeze(0)).squeeze(0)
    
    best_val_error = float('inf')
    
    # -- Precompute features for the validation set --
    val_features = []
    val_targets_list = []
    with torch.no_grad():
        feature_extractor.eval()
        for features, targets in val_dataset:
            # Single-sample batch dimension
            features = features.unsqueeze(0)
            sequence_length = features.size(1)
            num_windows = sequence_length - window_size + 1
            
            # Collect feature vectors for each sliding window in this sample
            window_features = []
            for t in range(num_windows):
                current_state = features[:, t:t+window_size, :]
                window_features.append(feature_extractor(current_state))
            
            # Combine into a single tensor, shape (num_windows, feature_dim)
            val_features.append(torch.stack(window_features, dim=0).squeeze(1))
            # Assume last target is the relevant RUL
            val_targets_list.append(targets[-1].item())
    
    print("Starting Linear TD(λ) training with batch_size =", batch_size)
    
    # -- Training loop --
    for epoch in range(epochs):
        total_loss = 0.0
        feature_extractor.eval()  # Not training the CNN
        value_function.train()    # Train the linear head
        
        for batch_features, batch_targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch_features = batch_features.to(feature_extractor.device)
            batch_targets  = batch_targets.to(value_function.device)
            
            B = batch_features.size(0)
            sequence_length = batch_features.size(1)
            num_windows = sequence_length - window_size + 1
            if num_windows < 1:
                continue
            
            # 1) Precompute features for each sliding window in this batch
            precomputed_features = []
            with torch.no_grad():
                for t in range(num_windows):
                    current_state = batch_features[:, t : t + window_size, :]
                    features_out = feature_extractor(current_state)
                    precomputed_features.append(features_out)  # shape: (B, feature_dim)
            
            # 2) Initialize eligibility traces for each parameter
            eligibility_traces = [
                torch.zeros((B,) + p.shape, device=value_function.device) 
                for p in params
            ]
            
            # 3) Get initial value from the first window to use in the update
            v_old = f_value_function(params, precomputed_features[0])  # shape: [B]
            episode_loss = 0.0
            
            # 4) TD(λ) through the sliding windows
            for t in range(num_windows):
                current_features = precomputed_features[t]        # shape (B, feature_dim)
                v = f_value_function(params, current_features)    # shape (B,)
                
                # Compute reward from the target at the last time step in the current window
                current_rul = batch_targets[:, t + window_size - 1]
                if use_piecewise:
                    reward = torch.where(
                        current_rul > early_rul,
                        torch.tensor(0.0, device=value_function.device),
                        torch.tensor(1.0, device=value_function.device)
                    )
                else:
                    reward = torch.ones_like(current_rul, device=value_function.device)
                
                # Compute next value if not at the final window
                if t < num_windows - 1:
                    next_features = precomputed_features[t+1]
                    v_next = f_value_function(params, next_features)
                else:
                    v_next = torch.zeros_like(v)
                
                # TD error: δ = r + γ*V(next) - V(current)
                delta = reward + gamma * v_next - v  # shape: [B]
                
                # Per-sample gradients: grad of the single-sample value
                per_sample_grad_fn = vmap(grad(single_sample_value), in_dims=(None, 0))
                grads = per_sample_grad_fn(params, current_features)
                # 'grads' is a tuple of tensors matching 'params', each of shape [B, *param.shape]
                
                # The paper "True Online TD(λ)" uses these additional terms:
                factor = learning_rate * (delta + (v - v_old))  # shape: [B]
                factor2 = learning_rate * (v - v_old)           # shape: [B]
                
                updates = []
                new_eligibility_traces = []
                for e_trace, g, p in zip(eligibility_traces, grads, params):
                    # e_trace shape: [B, *param.shape]
                    # g shape:       [B, *param.shape]
                    
                    # dot product across each sample in the batch
                    dot_prod = torch.sum(e_trace.view(B, -1) * g.view(B, -1), dim=1)  # shape: [B]
                    
                    # New eligibility trace:
                    # e_{t} = γλ e_{t-1} + (1 - αγλ dot_prod) * g
                    new_e = (gamma * lambd * e_trace
                             + (1 - learning_rate * gamma * lambd * dot_prod).view(B, *([1] * len(p.shape))) * g)
                    new_eligibility_traces.append(new_e)
                    
                    # Parameter update = factor * new_e - factor2 * g, aggregated over the batch
                    update_i = factor.view(B, *([1]*len(p.shape))) * new_e \
                               - factor2.view(B, *([1]*len(p.shape))) * g
                    updates.append(update_i.mean(dim=0))
                
                eligibility_traces = new_eligibility_traces
                
                # Apply updates to params
                params = tuple(p + u for p, u in zip(params, updates))
                
                # Copy updated params into the actual model
                with torch.no_grad():
                    for orig_p, new_p in zip(value_function.parameters(), params):
                        orig_p.copy_(new_p)
                
                # v_old <- v_next for next iteration
                v_old = v_next.detach()
                
                # Track loss (MSE of TD error)
                episode_loss += torch.mean(delta**2).item()
            
            total_loss += episode_loss
        
        # Average training loss across the entire dataset
        avg_train_loss = total_loss / len(train_loader.dataset)
        
        # -- Validation --
        value_function.eval()
        val_preds = []
        with torch.no_grad():
            # For each engine in validation, use the last window's features to predict
            for features in val_features:
                val_pred = value_function(features[-1]).item()
                val_preds.append(val_pred)
        
        true_val_error = np.sqrt(mean_squared_error(val_targets_list, val_preds))
        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val RUL Error (RMSE): {true_val_error:.4f}")
        
        # Save the best-performing model
        if true_val_error < best_val_error:
            best_val_error = true_val_error
            os.makedirs(model_save_path, exist_ok=True)
            torch.save({
                'feature_extractor': feature_extractor.state_dict(),
                'value_function': value_function.state_dict()
            }, os.path.join(model_save_path, f"best_{model_name}.pth"))
            print("Best Model Saved")
    
    # Load best model before returning
    checkpoint = torch.load(os.path.join(model_save_path, f"best_{model_name}.pth"))
    feature_extractor.load_state_dict(checkpoint['feature_extractor'])
    value_function.load_state_dict(checkpoint['value_function'])
    
    return feature_extractor, value_function


def main():
    """
    Main function demonstrating the TD(λ) algorithm.
    It:
      - Loads and preprocesses the CMAPSS dataset.
      - Splits the data into windows for each engine.
      - Trains a CNN-based feature extractor + linear head model with TD(λ).
      - Saves the best model checkpoint.
    """
    # Prepare data with explicit parameters
    train_data, test_data, true_rul, sequence_length, shift, early_rul = prepare_data(
        train_path="../CMAPSSData/train_FD001.txt",
        test_path="../CMAPSSData/test_FD001.txt",
        rul_path="../CMAPSSData/RUL_FD001.txt",
        window_length=200,
        shift=1,
        early_rul=125
    )
    
    # Build lists to collect windowed data across all engines
    processed_train_data = []
    processed_train_targets = []
    num_train_machines = len(train_data[0].unique())
    
    for i in range(1, num_train_machines + 1):
        # Extract sensor columns for engine i
        temp_train_data = train_data[train_data[0] == i].drop(columns=[0]).values
        
        # Generate a target RUL array for this engine
        temp_train_targets = process_targets(
            data_length=temp_train_data.shape[0],
            early_rul=early_rul
        )
        
        # Create sliding windows from the entire sequence
        data_for_machine, targets_for_machine = process_input_data_with_targets(
            temp_train_data, 
            temp_train_targets, 
            window_length=sequence_length, 
            shift=shift, 
            target_for_all_timestamps=True
        )
        
        processed_train_data.append(data_for_machine)
        processed_train_targets.append(targets_for_machine)
    
    # Concatenate data from all engines
    processed_train_data = np.concatenate(processed_train_data)
    processed_train_targets = np.concatenate(processed_train_targets)
    
    print(f"Processed Train Data Shape: {processed_train_data.shape}")
    print(f"Processed Train Targets Shape: {processed_train_targets.shape}")
    
    # Train the model via Linear TD(λ)
    feature_extractor, value_function = train_linear_td_lambda_model(
        train_data=processed_train_data, 
        train_targets=processed_train_targets, 
        batch_size=32,
        model_name="linear_td_lambda",
        use_piecewise=True,
        early_rul=early_rul,
        window_size=50,  # Capture more context with a larger window
        gamma=0.99,
        lambd=0,
        learning_rate=0.1,
        epochs=20
    )
    
if __name__ == "__main__":
    print("Running TD algorithm main() with Linear TD(λ)")
    main()
