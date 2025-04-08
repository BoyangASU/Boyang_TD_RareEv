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
    Computes the root mean squared error (RMSE) on a validation dataset using 
    the given model and a specified window size for inference.

    Args:
        model (torch.nn.Module): Model to evaluate.
        val_dataset (torch.utils.data.Dataset): Validation dataset.
        window_size (int): The size of the time window used for the final prediction.

    Returns:
        float: RMSE between model predictions and ground truth RUL on the entire validation set.
    """
    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        # Loop through each sample (feature/target pair) in the validation dataset
        for features, targets in val_dataset:
            # Expand dimension to (batch=1, seq_len, features)
            features = features.unsqueeze(0).to(model.device)
            targets = targets.unsqueeze(0).to(model.device)

            # Determine how many windows can be formed from the time sequence
            sequence_length = features.size(1)
            num_windows = sequence_length - window_size + 1

            # Use only the last window for making a prediction
            last_window = features[:, num_windows - 1 : num_windows - 1 + window_size, :]
            pred = model(last_window).squeeze().detach().cpu().item()
            true_val = targets[:, -1].squeeze().detach().cpu().item()

            all_preds.append(pred)
            all_true.append(true_val)

    # Compute RMSE over the entire validation set
    return np.sqrt(mean_squared_error(all_true, all_preds))


class FeatureExtractorCNN(torch.nn.Module):
    """
    A 1D CNN feature extractor that takes a sequence of sensor readings and
    produces a fixed-length feature vector.

    Args:
        input_channels (int): Number of input channels (i.e., sensor features).
    
    Attributes:
        device (torch.device): Determines if we're running on CPU or CUDA.
        cnn_layers (torch.nn.Sequential): Sequence of Conv1D and ReLU layers for feature extraction.
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
        Forward pass through the CNN feature extractor.
        
        Args:
            x (torch.Tensor): Input data of shape (batch_size, seq_length, input_channels).

        Returns:
            torch.Tensor: Feature tensor of shape (batch_size, 64).
        """
        # Move channels to first dimension for PyTorch Conv1D: (batch, channels, seq_length)
        x = x.transpose(1, 2).to(self.device)
        # Use gradient checkpointing to reduce memory usage; the output shape is (batch, 64, seq_length).
        out = checkpoint(self.cnn_layers, x)
        # Pool (mean) across the seq_length dimension => (batch, 64)
        return out.mean(dim=2)


class LinearValueFunction(torch.nn.Module):
    """
    A simple linear head that takes the extracted feature vector and outputs a scalar 
    representing the predicted RUL (or value).

    Args:
        input_size (int): The dimension of the incoming feature vector.
    
    Attributes:
        device (torch.device): Identifies if we're running on CPU or GPU.
        linear (torch.nn.Linear): A fully connected layer mapping from input_size -> 1.
    """
    def __init__(self, input_size=64):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.linear = torch.nn.Linear(input_size, 1).to(self.device)
        
    def forward(self, x):
        """
        Forward pass to generate a scalar prediction from the feature vector.
        
        Args:
            x (torch.Tensor): Feature tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Scalar prediction (batch_size,).
        """
        return self.linear(x).squeeze()


class CombinedModel(torch.nn.Module):
    """
    A combined model that wraps the CNN feature extractor and the linear value function,
    making it easier to save, load, and evaluate a single model instance.

    Args:
        feature_extractor (FeatureExtractorCNN): CNN for feature extraction.
        value_function (LinearValueFunction): Linear head that outputs the RUL (or value).
    """
    def __init__(self, feature_extractor, value_function):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.value_function = value_function
        self.device = feature_extractor.device
        
    def forward(self, x):
        """
        Passes the input through the feature extractor, then the value function.

        Args:
            x (torch.Tensor): Shape (batch_size, seq_length, input_channels).

        Returns:
            torch.Tensor: Shape (batch_size,) containing scalar predictions.
        """
        return self.value_function(self.feature_extractor(x))


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
    Trains a CNN-based feature extractor + linear head using manual Temporal Difference (TD) updates with λ,
    without using a standard optimizer. All parameter updates rely on eligibility traces.

    Args:
        train_data (np.ndarray): Training data array of shape (num_samples, seq_len, num_channels).
        train_targets (np.ndarray): Targets (RUL) of shape (num_samples, seq_len).
        batch_size (int): Batch size for data loading.
        epochs (int): Number of epochs.
        learning_rate (float): The scalar step size for updates.
        model_save_path (str): Directory to save model checkpoints.
        model_name (str): Filename prefix for saved models.
        use_piecewise (bool): Whether to apply a piecewise reward function (0 if RUL>early_rul else 1).
        early_rul (int): The RUL threshold for piecewise rewards.
        gamma (float): Discount factor for TD updates.
        lambd (float): Lambda parameter for eligibility traces.
        window_size (int): Size of the sliding window for each input subsequence.

    Returns:
        (FeatureExtractorCNN, LinearValueFunction, float): The trained feature extractor, 
        the value head, and the best validation RMSE achieved.
    """
    # Clean up GPU memory if applicable
    torch.cuda.empty_cache()
    gc.collect()

    from sklearn.model_selection import train_test_split
    from train import RULDataset  # Ensure this import matches your project structure

    # -----------------------------
    # 1. Split train/val data
    # -----------------------------
    train_data, val_data, train_targets, val_targets = train_test_split(
        train_data, train_targets, test_size=0.2, random_state=83
    )
    
    # Convert into PyTorch datasets
    train_dataset = RULDataset(train_data, train_targets)
    val_dataset = RULDataset(val_data, val_targets)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    # -----------------------------
    # 2. Model initialization
    # -----------------------------
    feature_extractor = FeatureExtractorCNN(input_channels=train_data.shape[2])
    value_function = LinearValueFunction(input_size=64)

    # Weight initialization
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

    # (Optional) freeze some CNN layers (e.g., earlier layers)
    for param in feature_extractor.parameters():
        param.requires_grad = False
    # Unfreeze just the last layers in the CNN
    for layer in feature_extractor.cnn_layers[4:]:
        for param in layer.parameters():
            param.requires_grad = True

    # Collect only trainable parameters
    trainable_params = []
    for p in feature_extractor.parameters():
        if p.requires_grad:
            trainable_params.append(p)
    for p in value_function.parameters():
        trainable_params.append(p)

    best_val_error = float('inf')

    # -----------------------------
    # 3. Training Loop
    # -----------------------------
    for epoch in range(epochs):
        feature_extractor.train()
        value_function.train()
        total_loss = 0.0

        # Iterate over training batches
        for batch_features, batch_targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            # Move batch data to the device
            batch_features = batch_features.to(feature_extractor.device, non_blocking=True)
            batch_targets = batch_targets.to(value_function.device, non_blocking=True)

            sequence_length = batch_features.size(1)
            num_windows = sequence_length - window_size + 1
            if num_windows < 1:
                continue

            # Initialize eligibility traces for each trainable parameter
            eligibility_traces = []
            for p in trainable_params:
                if len(p.shape) == 1:
                    # Bias terms have shape (param_dim,)
                    eligibility_traces.append(torch.zeros_like(p, device=p.device))
                else:
                    # Weight matrices have shape (batch_size, *param_dims)
                    eligibility_traces.append(torch.zeros(
                        (batch_features.shape[0],) + p.shape, device=p.device
                    ))

            # Process windows in smaller chunks to save memory
            chunk_size = min(10, num_windows)
            for chunk_start in range(0, num_windows, chunk_size):
                chunk_end = min(chunk_start + chunk_size, num_windows)

                for t in range(chunk_start, chunk_end):
                    # Current window of data and corresponding RUL
                    current_window = batch_features[:, t : t + window_size, :]
                    current_rul = batch_targets[:, t + window_size - 1]

                    # Forward pass: compute current value estimate
                    v = value_function(feature_extractor(current_window))

                    # Compute reward (piecewise or constant)
                    reward = (torch.where(current_rul > early_rul, 0.0, 1.0) 
                              if use_piecewise else torch.ones_like(current_rul))

                    # Compute next value, if not at the end
                    if t < num_windows - 1:
                        next_window = batch_features[:, t + 1 : t + 1 + window_size, :]
                        v_next = value_function(feature_extractor(next_window))
                    else:
                        v_next = torch.zeros_like(v)

                    # TD error: δ = r + γ * V(next) - V(current)
                    delta = reward + gamma * v_next - v

                    if torch.isnan(delta).any():
                        print("Warning: NaN in delta, skipping update")
                        continue

                    # Compute gradients of V(current) wrt trainable params
                    grads = torch.autograd.grad(
                        v.sum(),    # sum for a combined gradient over the batch
                        trainable_params,
                        create_graph=True,
                        allow_unused=True
                    )

                    # Update eligibility traces & parameters
                    with torch.no_grad():
                        for i, (param, grad) in enumerate(zip(trainable_params, grads)):
                            if grad is None:
                                # Some parameters may not receive gradients
                                continue

                            # Optional gradient clipping
                            grad = torch.clamp(grad, -1.0, 1.0)

                            # For biases
                            if len(param.shape) == 1:
                                # e(t) = γ * λ * e(t-1) + mean(grad over batch)
                                eligibility_traces[i] = gamma * lambd * eligibility_traces[i] + grad.mean(dim=0)
                                # Δparam = α * mean(δ) * e(t)
                                param_update = learning_rate * delta.mean() * eligibility_traces[i]
                            else:
                                # For weights
                                eligibility_traces[i] = gamma * lambd * eligibility_traces[i] + grad
                                # Multiply each sample's δ to its trace, then average
                                reshape_dims = (delta.shape[0],) + (1,) * (len(param.shape))
                                delta_reshaped = delta.view(reshape_dims)
                                param_update = learning_rate * delta_reshaped * eligibility_traces[i]
                                param_update = param_update.mean(dim=0)

                            # Apply update if not NaN
                            if not torch.isnan(param_update).any():
                                param.add_(param_update)

                    # Accumulate MSE of TD error
                    total_loss += torch.mean(delta**2).item()

                    # Free memory from large tensors
                    del current_window, v, delta, grads, v_next, reward
                    torch.cuda.empty_cache()

        # -----------------------------
        # 4. Validation
        # -----------------------------
        combined_model = CombinedModel(feature_extractor, value_function)
        combined_model.eval()
        val_preds = []
        val_targets_list = []

        with torch.no_grad():
            # Evaluate on validation set by always taking the last window
            for features, targets in val_dataset:
                features = features.unsqueeze(0).to(feature_extractor.device)
                sequence_length = features.size(1)
                num_windows = sequence_length - window_size + 1
                if num_windows < 1:
                    continue

                last_window = features[:, num_windows - 1 : num_windows - 1 + window_size, :]
                pred = combined_model(last_window).squeeze().cpu().item()

                # Handle any NaN predictions
                if np.isnan(pred):
                    pred = 0.0
                val_preds.append(pred)
                val_targets_list.append(targets[-1].item())

        # Replace NaNs with 0 (final safety check)
        val_preds = np.nan_to_num(val_preds, nan=0.0)
        val_error = np.sqrt(mean_squared_error(val_targets_list, val_preds))
        avg_train_loss = total_loss / len(train_loader)

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {avg_train_loss:.4f}, Val RMSE: {val_error:.4f}"
        )

        # Save the best model parameters
        if val_error < best_val_error:
            best_val_error = val_error
            os.makedirs(model_save_path, exist_ok=True)
            torch.save({
                'feature_extractor': feature_extractor.state_dict(),
                'value_function': value_function.state_dict()
            }, os.path.join(model_save_path, f"best_{model_name}.pth"))
            print("Best Model Saved")

    # -----------------------------
    # 5. Load Best Model
    # -----------------------------
    checkpoint_data = torch.load(os.path.join(model_save_path, f"best_{model_name}.pth"))
    feature_extractor.load_state_dict(checkpoint_data['feature_extractor'])
    value_function.load_state_dict(checkpoint_data['value_function'])

    # Return the fully trained (best) feature extractor, value function, and the best validation RMSE
    return feature_extractor, value_function, best_val_error


def main():
    """
    Main execution function that:
      1. Prepares data from CMAPSS files.
      2. Creates train/val splits for multiple engines.
      3. Trains multiple TD(λ) models across various (lambda, gamma) settings.
      4. Prints a LaTeX-formatted table showing RMSE for each combination.
      5. Reports the best performing (lambda, gamma) and its RMSE.
    """
    # 1. Data Preparation
    train_data, test_data, true_rul, sequence_length, shift, early_rul = prepare_data(
        train_path="../CMAPSSData/train_FD001.txt",
        test_path="../CMAPSSData/test_FD001.txt",
        rul_path="../CMAPSSData/RUL_FD001.txt",
        window_length=200,
        shift=1,
        early_rul=125
    )
    
    # Create lists to hold windowed data and targets for each engine
    processed_train_data = []
    processed_train_targets = []
    num_train_machines = len(train_data[0].unique())

    # 2. Construct sliding windows for each engine
    for i in range(1, num_train_machines + 1):
        # Separate the rows for engine i, ignoring the engine ID column
        temp_train_data = train_data[train_data[0] == i].drop(columns=[0]).values

        # Generate the target RUL values for this engine's data
        temp_train_targets = process_targets(
            data_length=temp_train_data.shape[0],
            early_rul=early_rul
        )
        
        # Create input windows and aligned targets
        data_for_machine, targets_for_machine = process_input_data_with_targets(
            temp_train_data, 
            temp_train_targets, 
            window_length=sequence_length, 
            shift=shift, 
            target_for_all_timestamps=True
        )
        
        processed_train_data.append(data_for_machine)
        processed_train_targets.append(targets_for_machine)
    
    # Concatenate data from all engines into single arrays
    processed_train_data = np.concatenate(processed_train_data)
    processed_train_targets = np.concatenate(processed_train_targets)

    print(f"Processed Train Data Shape: {processed_train_data.shape}")
    print(f"Processed Train Targets Shape: {processed_train_targets.shape}")

    # 3. Define hyperparameter ranges for λ and γ
    lambda_values = [0.0, 0.2, 0.5, 0.9]
    gamma_values = [0.95, 0.99, 0.999, 1.0]
    
    # We will store the final RMSE of the best epoch for each (λ, γ) pair
    rmse_results = []
    
    # Track overall best model performance
    best_overall_rmse = float('inf')
    best_params = (None, None)
    
    # 4. Train a model for each combination of (λ, γ)
    for lam in lambda_values:
        row_results = []
        for gam in gamma_values:
            print(f"\n=== Training with lambda={lam}, gamma={gam} ===")
            
            # Train the model with given hyperparameters
            _, _, best_val_error = train_linear_td_lambda_model(
                train_data=processed_train_data,
                train_targets=processed_train_targets,
                batch_size=64,
                epochs=20,
                learning_rate=0.01,
                model_save_path="saved_models",
                model_name=f"td_lambda_lam{lam}_gam{gam}",
                use_piecewise=False,
                early_rul=early_rul,
                gamma=gam,
                lambd=lam,
                window_size=50  # example window size
            )
            
            # Append the best RMSE for this (λ, γ) to the row results
            row_results.append(best_val_error)
            
            # Update best overall performance if needed
            if best_val_error < best_overall_rmse:
                best_overall_rmse = best_val_error
                best_params = (lam, gam)
        
        rmse_results.append(row_results)

    # 5. Produce a LaTeX-friendly table of results
    print("\nLaTeX table of RMSE results (lambda in rows, gamma in columns):")
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\caption{RMSE for Different Settings of $\\lambda$ and $\\gamma$ with learning rate 0.1}")
    print("\\label{tab:rmse_matrix}")
    print("\\begin{tabular}{l|cccc}")
    print("\\hline")
    print(" & \\multicolumn{4}{c}{\\textbf{Gamma}} \\\\")
    print("$\\lambda$ & 0.95 & 0.99 & 0.999 & 1.00 \\\\")
    print("\\hline")
    
    # Fill each row with RMSE values
    for i, lam in enumerate(lambda_values):
        row_str = f"{lam}"
        for j, gam in enumerate(gamma_values):
            row_str += f" & {rmse_results[i][j]:.2f}"
        row_str += " \\\\"
        print(row_str)
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")
    
    # Report the best-performing hyperparameter pair
    print(f"\nBest overall RMSE: {best_overall_rmse:.2f} at lambda={best_params[0]}, gamma={best_params[1]}")

if __name__ == "__main__":
    print("Running TD(λ) experiments with multiple lambda/gamma settings.")
    main()
