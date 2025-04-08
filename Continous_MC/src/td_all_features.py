import torch
import numpy as np
from torch.utils.data import DataLoader
from train import CNN1D, RULDataset
from preprocessing import prepare_data, process_input_data_with_targets, process_targets
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import gc
from torch.utils.checkpoint import checkpoint

def compute_true_val_error(model, val_dataset, window_size):
    """
    Compute validation RMSE for a given model using sliding windows.
    
    Args:
        model (torch.nn.Module): Trained model to evaluate.
        val_dataset (torch.utils.data.Dataset): Dataset for validation.
        window_size (int): Size of the input window used by the model.
    
    Returns:
        float: RMSE between predictions and ground truth labels on the validation set.
    """
    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        # Iterate over each data/label pair in the validation dataset
        for features, targets in val_dataset:
            # Wrap the feature/target in a batch dimension for model processing
            features = features.unsqueeze(0).to(model.device)
            targets = targets.unsqueeze(0).to(model.device)

            # Compute how many windows we can create from the current sample
            sequence_length = features.size(1)
            num_windows = sequence_length - window_size + 1

            # Take the last window from this sequence for inference
            last_window = features[:, num_windows - 1:num_windows - 1 + window_size, :]
            
            # Predict RUL from the last window
            pred = model(last_window).squeeze().detach().cpu().item()

            # Ground truth is the last element of the target sequence
            true_val = targets[:, -1].squeeze().detach().cpu().item()

            # Collect predictions and ground truths for later RMSE calculation
            all_preds.append(pred)
            all_true.append(true_val)
    
    # Compute and return the RMSE
    return np.sqrt(mean_squared_error(all_true, all_preds))


class FeatureExtractorCNN(torch.nn.Module):
    """
    CNN-based feature extractor: transforms 1D sensor data into a latent feature representation.

    Args:
        input_channels (int): Number of input channels/features.
    
    Attributes:
        device (torch.device): Identifies if we're running on CPU or GPU.
        cnn_layers (torch.nn.Sequential): Stacked convolution + ReLU blocks for feature extraction.
    """
    def __init__(self, input_channels):
        super().__init__()
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
        """
        Forward pass for feature extraction.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_channels).
        
        Returns:
            torch.Tensor: Aggregated feature vector of shape (batch_size, 64).
        """
        # Transpose to (batch_size, input_channels, seq_length) for Conv1D
        x = x.transpose(1, 2).to(self.device)
        # Use gradient checkpointing to reduce memory usage; output is shape (batch_size, 64, seq_length)
        out = checkpoint(self.cnn_layers, x)
        # Pool across the seq_length dimension by taking the mean
        return out.mean(dim=2)


class LinearValueFunction(torch.nn.Module):
    """
    A simple linear head for mapping extracted features to a scalar RUL prediction.
    
    Args:
        input_size (int): Dimensionality of the incoming feature vector.
    
    Attributes:
        device (torch.device): Identifies if we're running on CPU or GPU.
        linear (torch.nn.Linear): A fully-connected layer that outputs a single value.
    """
    def __init__(self, input_size=64):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.linear = torch.nn.Linear(input_size, 1).to(self.device)
        
    def forward(self, x):
        """
        Forward pass for the value function head.
        
        Args:
            x (torch.Tensor): Feature tensor of shape (batch_size, input_size).
        
        Returns:
            torch.Tensor: Scalar prediction (batch_size,).
        """
        return self.linear(x).squeeze()


class CombinedModel(torch.nn.Module):
    """
    Combines the feature extractor and the value function into a single model.
    
    Args:
        feature_extractor (FeatureExtractorCNN): CNN-based feature extractor.
        value_function (LinearValueFunction): Linear module mapping features to predictions.
    """
    def __init__(self, feature_extractor, value_function):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.value_function = value_function
        self.device = feature_extractor.device
        
    def forward(self, x):
        """
        Pass the input through the feature extractor, then the value function.

        Args:
            x (torch.Tensor): Input sensor data (batch_size, seq_length, input_channels).
        
        Returns:
            torch.Tensor: Scalar RUL predictions for the batch (batch_size,).
        """
        features = self.feature_extractor(x)
        return self.value_function(features)


def train_linear_td_lambda_model(
    train_data: np.ndarray,
    train_targets: np.ndarray,
    batch_size: int = 32,
    epochs: int = 20,
    learning_rate: float = 0.01,
    model_save_path: str = "saved_models",
    model_name: str = "linear_td_lambda",
    use_piecewise: bool = True,
    early_rul: int = 125,
    gamma: float = 0.99,
    lambd: float = 0.0,
    window_size: int = 30
) -> tuple:
    """
    Trains a CNN-based feature extractor and a linear value head using a Temporal Difference (TD) method with λ.
    Instead of using an optimizer, parameter updates happen manually based on TD error and eligibility traces.

    Args:
        train_data (np.ndarray): Training input data of shape (num_samples, seq_length, num_channels).
        train_targets (np.ndarray): Corresponding target RUL data of shape (num_samples, seq_length).
        batch_size (int): Batch size for loading the training set.
        epochs (int): Number of epochs to train.
        learning_rate (float): Learning rate for manual TD updates.
        model_save_path (str): Directory in which to save model checkpoints.
        model_name (str): Name prefix for saved model files.
        use_piecewise (bool): Whether to use piecewise reward (0 if RUL>early_rul else 1) or constant reward (1).
        early_rul (int): Threshold for piecewise reward function.
        gamma (float): Discount factor for TD learning.
        lambd (float): λ parameter for eligibility traces.
        window_size (int): Size of the temporal window used by the model.

    Returns:
        (FeatureExtractorCNN, LinearValueFunction): The best-performing model components after training.
    """
    # Cleanup to free GPU memory if available
    torch.cuda.empty_cache()
    gc.collect()

    # Split data into training and validation sets
    train_data, val_data, train_targets, val_targets = train_test_split(
        train_data, train_targets, test_size=0.2, random_state=83
    )
    
    # Convert numpy arrays to RULDataset objects for PyTorch Dataloader
    train_dataset = RULDataset(train_data, train_targets)
    val_dataset = RULDataset(val_data, val_targets)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    # Create model components: CNN feature extractor + linear head
    feature_extractor = FeatureExtractorCNN(input_channels=train_data.shape[2])
    value_function = LinearValueFunction(input_size=64)

    # Weight initialization for better convergence
    for m in feature_extractor.modules():
        if isinstance(m, torch.nn.Conv1d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    for m in value_function.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    # Optional: Freeze the first few CNN layers to reduce complexity
    for param in feature_extractor.parameters():
        param.requires_grad = False
    # Unfreeze the last few layers of the CNN
    for layer in feature_extractor.cnn_layers[4:]:
        for param in layer.parameters():
            param.requires_grad = True

    # Gather a list of trainable parameters (unfrozen CNN + entire linear head)
    trainable_params = []
    for p in feature_extractor.parameters():
        if p.requires_grad:
            trainable_params.append(p)
    for p in value_function.parameters():
        trainable_params.append(p)

    best_val_error = float('inf')

    # ---------------------------------
    # Training loop with manual TD(λ)
    # ---------------------------------
    for epoch in range(epochs):
        feature_extractor.train()
        value_function.train()
        total_loss = 0.0

        # We don't use standard optimizer steps. We compute TD error manually and update weights accordingly.
        for batch_features, batch_targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch_features = batch_features.to(feature_extractor.device, non_blocking=True)
            batch_targets = batch_targets.to(value_function.device, non_blocking=True)

            # Calculate how many windows we can form from the current batch
            sequence_length = batch_features.size(1)
            num_windows = sequence_length - window_size + 1
            if num_windows < 1:
                continue

            # Initialize eligibility traces for each parameter
            # We track them across the batch dimension so each sample has its own trace
            eligibility_traces = []
            for p in trainable_params:
                if len(p.shape) == 1:
                    # Bias vector: create a single eligibility trace with same shape
                    eligibility_traces.append(torch.zeros_like(p, device=p.device))
                else:
                    # Weight matrix: create a trace with batch dimension plus param shape
                    eligibility_traces.append(torch.zeros(
                        (batch_features.shape[0],) + p.shape, device=p.device
                    ))

            # Process windows in chunks (to reduce memory usage)
            chunk_size = min(10, num_windows)
            for chunk_start in range(0, num_windows, chunk_size):
                chunk_end = min(chunk_start + chunk_size, num_windows)

                for t in range(chunk_start, chunk_end):
                    current_window = batch_features[:, t:t+window_size, :]
                    current_rul = batch_targets[:, t+window_size-1]

                    # Compute current value estimates
                    v = value_function(feature_extractor(current_window))

                    # Reward can be piecewise or constant
                    reward = torch.where(
                        current_rul > early_rul, 0.0, 1.0
                    ) if use_piecewise else torch.ones_like(current_rul)

                    # Estimate future value
                    if t < num_windows - 1:
                        next_window = batch_features[:, t+1 : t+1+window_size, :]
                        v_next = value_function(feature_extractor(next_window))
                    else:
                        # No next state at the end of the sequence
                        v_next = torch.zeros_like(v)

                    # Calculate TD error: δ = r + γ * V(next) - V(current)
                    delta = reward + gamma * v_next - v

                    if torch.isnan(delta).any():
                        print("Warning: NaN in delta, skipping update")
                        continue

                    # Compute gradients of V(current) wrt to each trainable parameter
                    grads = torch.autograd.grad(
                        v.sum(),    
                        trainable_params,
                        create_graph=True,
                        allow_unused=True
                    )

                    # Manually update eligibility traces and parameters
                    with torch.no_grad():
                        for i, (param, grad) in enumerate(zip(trainable_params, grads)):
                            if grad is None:
                                continue

                            # (Optional) gradient clipping
                            grad = torch.clamp(grad, -1.0, 1.0)

                            # Update eligibility traces
                            if len(param.shape) == 1:
                                # Bias term: 
                                # e(t) = γ * λ * e(t-1) + mean of grad across batch
                                eligibility_traces[i] = gamma * lambd * eligibility_traces[i] + grad.mean(dim=0)
                                param_update = learning_rate * delta.mean() * eligibility_traces[i]
                            else:
                                # Weight term: track trace for each item in the batch
                                eligibility_traces[i] = gamma * lambd * eligibility_traces[i] + grad

                                # Multiply each sample's delta into its eligibility trace, then average
                                reshape_dims = (delta.shape[0],) + (1,) * (len(param.shape))
                                delta_reshaped = delta.view(reshape_dims)
                                param_update = learning_rate * delta_reshaped * eligibility_traces[i]
                                param_update = param_update.mean(dim=0)

                            # Apply parameter updates only if there's no NaN
                            if not torch.isnan(param_update).any():
                                param.add_(param_update)

                    # Track MSE of the TD error
                    total_loss += torch.mean(delta**2).item()

                    # Cleanup
                    del current_window, v, delta, grads, v_next, reward
                    torch.cuda.empty_cache()

        # -----------------------------
        # Validation phase
        # -----------------------------
        combined_model = CombinedModel(feature_extractor, value_function)
        combined_model.eval()
        val_preds = []
        val_targets_list = []

        # Inference on the validation dataset
        with torch.no_grad():
            for features, targets in val_dataset:
                features = features.unsqueeze(0).to(feature_extractor.device)
                seq_len = features.size(1)
                num_windows_val = seq_len - window_size + 1
                if num_windows_val < 1:
                    continue

                # Use the last window from the validation sequence
                last_window = features[:, num_windows_val - 1 : num_windows_val - 1 + window_size, :]
                pred = combined_model(last_window).squeeze().cpu().item()

                # Guard against NaN predictions
                if np.isnan(pred):
                    pred = 0.0
                val_preds.append(pred)
                val_targets_list.append(targets[-1].item())

        # Replace any NaNs in predictions with 0
        val_preds = np.nan_to_num(val_preds, nan=0.0)
        val_error = np.sqrt(mean_squared_error(val_targets_list, val_preds))
        avg_train_loss = total_loss / len(train_loader)

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {avg_train_loss:.4f}, Val RMSE: {val_error:.4f}"
        )

        # Save the best model checkpoint
        if val_error < best_val_error:
            best_val_error = val_error
            os.makedirs(model_save_path, exist_ok=True)
            torch.save({
                'feature_extractor': feature_extractor.state_dict(),
                'value_function': value_function.state_dict()
            }, os.path.join(model_save_path, f"best_{model_name}.pth"))
            print("Best Model Saved")

    # -----------------------------
    # Final: load the best weights
    # -----------------------------
    checkpoint_data = torch.load(os.path.join(model_save_path, f"best_{model_name}.pth"))
    feature_extractor.load_state_dict(checkpoint_data['feature_extractor'])
    value_function.load_state_dict(checkpoint_data['value_function'])

    return feature_extractor, value_function


def main():
    """
    Main function orchestrating data preparation, model training, and memory cleanup.
    """
    # Free GPU memory where possible
    torch.cuda.empty_cache()
    gc.collect()
    
    # Load and prepare data. Returns training data, test data, true RUL values, and other parameters.
    train_data, test_data, true_rul, sequence_length, shift, early_rul = prepare_data(
        train_path="../CMAPSSData/train_FD001.txt",
        test_path="../CMAPSSData/test_FD001.txt",
        rul_path="../CMAPSSData/RUL_FD001.txt",
        window_length=200,
        shift=1,
        early_rul=125
    )
    
    # Process data in chunks to avoid excessive memory usage
    processed_train_data = []
    processed_train_targets = []
    num_train_machines = len(train_data[0].unique())

    for i in range(1, num_train_machines + 1):
        temp_train_data = train_data[train_data[0] == i].drop(columns=[0]).values
        temp_train_targets = process_targets(temp_train_data.shape[0], early_rul)
        data, targets = process_input_data_with_targets(
            temp_train_data, temp_train_targets, 
            window_length=sequence_length, 
            shift=shift, 
            target_for_all_timestamps=True
        )
        processed_train_data.append(data)
        processed_train_targets.append(targets)
        
        # Periodic cleanup
        if i % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    # Combine the data arrays from all machines
    processed_train_data = np.concatenate(processed_train_data)
    processed_train_targets = np.concatenate(processed_train_targets)
    
    print(f"Processed Train Data Shape: {processed_train_data.shape}")
    print(f"Processed Train Targets Shape: {processed_train_targets.shape}")
    
    # Train the TD(λ) model with user-defined hyperparameters
    feature_extractor, value_function = train_linear_td_lambda_model(
        processed_train_data, 
        processed_train_targets, 
        batch_size=64,
        model_name="linear_td_lambda",
        use_piecewise=True,
        early_rul=early_rul,
        window_size=30,
        gamma=0.99,
        lambd=0,
        learning_rate=0.01,
        epochs=20
    )

if __name__ == "__main__":
    print("Running memory-optimized TD algorithm with batch size 32")
    main()
