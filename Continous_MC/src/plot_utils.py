import matplotlib.pyplot as plt
import numpy as np
import torch
from train import CNN1D
from sklearn.metrics import mean_squared_error
import os
from preprocessing import prepare_data, process_test_data

def load_model(model_path: str, input_channels: int) -> CNN1D:
    """Load a saved model."""
    model = CNN1D(input_channels=input_channels)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def plot_predictions(true_rul: np.ndarray, pred_rul: np.ndarray, save_path: str = None, method_name: str = ""):
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
    if method_name:
        plt.title(f"True vs Predicted RUL (Sorted by True RUL) - {method_name}")
    else:
        plt.title("True vs Predicted RUL (Sorted by True RUL)")
    plt.legend()
    plt.grid(True)
    
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    plt.show()

def compute_s_score(rul_true: np.ndarray, rul_pred: np.ndarray) -> float:
    """
    Compute the S-score metric for RUL prediction.
    
    Both rul_true and rul_pred should be 1D numpy arrays.
    """
    diff = rul_pred - rul_true
    return np.sum(np.where(diff < 0, np.exp(-diff/13)-1, np.exp(diff/10)-1))

def evaluate_predictions(true_rul: np.ndarray, pred_rul: np.ndarray):
    """Evaluate predictions using RMSE and S-score."""
    rmse = np.sqrt(mean_squared_error(true_rul, pred_rul))
    s_score = compute_s_score(true_rul, pred_rul)
    
    print(f"RMSE: {rmse:.4f}")
    print(f"S-score: {s_score:.4f}")
    
    return rmse, s_score

def plot_window(window_data: np.ndarray,
                window_rul: np.ndarray,
                sensor_names: list = None,
                mask: np.ndarray = None,
                title: str = "Sensor Signals and RUL for Window"):
    """
    Plot sensor signals and RUL for the provided window.
    
    Parameters:
        window_data (np.ndarray): Array of shape (window_length, num_features)
                                  with sensor signals.
        window_rul (np.ndarray): Array of shape (window_length,) containing RUL values.
        sensor_names (list, optional): List of sensor names corresponding to each feature.
                                       If None, sensors are numbered.
        mask (np.ndarray, optional): Binary mask of shape (window_length,) where valid time-steps
                                     are denoted by 1 and padded steps by 0.
        title (str): Title for the overall figure.
    
    Behavior:
        - If a mask is provided, the valid region (non-padded) is overlaid onto the full data.
          The full window is plotted with a dotted line (light color) while the valid portion is
          highlighted in a solid line with markers. A vertical dashed line indicates the transition.
        - If no mask is provided, the complete window is plotted.
    """
    window_length, num_features = window_data.shape
    time = np.arange(window_length)
    
    # Determine valid length from mask, if applicable.
    if mask is not None:
        valid_length = int(np.sum(mask))
    else:
        valid_length = window_length
    
    # Create subplots: one for each sensor and one for RUL.
    total_plots = num_features + 1
    fig, axes = plt.subplots(total_plots, 1, figsize=(12, 3 * total_plots), sharex=True)
    
    # Plot each sensor's data.
    for sensor in range(num_features):
        ax = axes[sensor]
        # Plot the entire window in light dotted style.
        ax.plot(time, window_data[:, sensor], linestyle=":", color="gray", marker="o", alpha=0.5, label="Full Window")
        # Overlay valid portion if mask is provided.
        if mask is not None and valid_length > 0:
            valid_time = np.arange(valid_length)
            ax.plot(valid_time, window_data[:valid_length, sensor], linestyle="-", color="blue", marker="o", label="Valid Data")
            # Mark the boundary between valid and padded.
            ax.axvline(valid_length - 1, linestyle="--", color="red")
        else:
            ax.plot(time, window_data[:, sensor], linestyle="-", color="blue", marker="o")
        if sensor_names and sensor < len(sensor_names):
            ax.set_title(sensor_names[sensor])
        else:
            ax.set_title(f"Sensor {sensor + 1}")
        ax.set_ylabel("Value")
        ax.grid(True)
        ax.legend(loc="best", fontsize="small")
    
    # Plot RUL in the last subplot.
    ax_rul = axes[-1]
    ax_rul.plot(time, window_rul, linestyle=":", color="gray", marker="o", alpha=0.5, label="Full Window")
    if mask is not None and valid_length > 0:
        valid_time = np.arange(valid_length)
        ax_rul.plot(valid_time, window_rul[:valid_length], linestyle="-", color="red", marker="o", linewidth=2, label="Valid RUL")
        ax_rul.axvline(valid_length - 1, linestyle="--", color="black")
    else:
        ax_rul.plot(time, window_rul, linestyle="-", color="red", marker="o", linewidth=2)
    ax_rul.set_title("RUL")
    ax_rul.set_xlabel("Timestep")
    ax_rul.set_ylabel("RUL")
    ax_rul.grid(True)
    ax_rul.legend(loc="best", fontsize="small")
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def main():
    """Main function to load model, make predictions, and create plots."""
    # Create directories
    os.makedirs("figures", exist_ok=True)
    
    # Load and process test data
    _, test_data, true_rul, window_length, shift, _ = prepare_data(
        train_path="../CMAPSSData/train_FD001.txt",
        test_path="../CMAPSSData/test_FD001.txt",
        rul_path="../CMAPSSData/RUL_FD001.txt"
    )
    
    # Process test data
    processed_test_data = []
    num_test_windows_list = []
    
    num_test_machines = len(test_data[0].unique())
    for i in range(1, num_test_machines + 1):
        temp_test_data = test_data[test_data[0] == i].drop(columns=[0]).values
        test_data_for_engine, num_windows = process_test_data(
            temp_test_data, window_length=window_length, shift=shift, num_test_windows=1
        )
        processed_test_data.append(test_data_for_engine)
        num_test_windows_list.append(num_windows)
    
    processed_test_data = np.concatenate(processed_test_data)
    
    # Load model and make predictions
    model = load_model("saved_models/best_model.pth", input_channels=processed_test_data.shape[2])
    
    with torch.no_grad():
        test_tensor = torch.FloatTensor(processed_test_data).to(model.device)
        predictions = model(test_tensor).cpu().numpy()
    
    # Process predictions
    preds_for_each_engine = np.split(predictions, np.cumsum(num_test_windows_list)[:-1])
    indices_of_last_examples = np.cumsum(num_test_windows_list) - 1
    preds_for_last_example = np.concatenate(preds_for_each_engine)[indices_of_last_examples]
    
    # Evaluate and plot results
    evaluate_predictions(true_rul, preds_for_last_example)
    
    # Create and save plots
    plot_predictions(
        true_rul, 
        preds_for_last_example, 
        save_path="figures/rul_predictions_sorted.png",
        method_name="Best Model"
    )
    
    # Create additional plot without sorting
    plt.figure(figsize=(10, 6))
    plt.plot(true_rul, label="True RUL", color="red")
    plt.plot(preds_for_last_example, label="Predicted RUL", color="blue")
    plt.xlabel("Engine Index")
    plt.ylabel("RUL")
    plt.title("True vs Predicted RUL")
    plt.legend()
    plt.grid(True)
    plt.savefig("figures/rul_predictions_unsorted.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Create a dummy window.
    window_length = 30
    num_sensors = 5
    # Create random sensor data.
    dummy_sensor_data = np.random.random((window_length, num_sensors))
    # Create dummy RUL that decreases linearly.
    dummy_rul_data = np.linspace(100, 0, window_length)
    sensor_names = [f"Sensor {i+1}" for i in range(num_sensors)]
    
    print("Plotting without mask (full valid window)...")
    # Plot without any padding (mask not provided).
    plot_window(dummy_sensor_data, dummy_rul_data, sensor_names=sensor_names,
                title="Window Plot (No Padding)")
    
    # Now create a mask to simulate a padded window.
    # For example, let the first 20 timesteps be valid and the rest be padded.
    mask = np.zeros(window_length, dtype=np.uint8)
    mask[:20] = 1  # first 20 timesteps are valid
    
    # Simulate padded values: Set padded sensor values to a constant value (e.g., -1) and RUL to -1.
    padded_sensor_data = dummy_sensor_data.copy()
    padded_sensor_data[20:, :] = -1
    padded_rul_data = dummy_rul_data.copy()
    padded_rul_data[20:] = -1

    print("Plotting with mask (padded window)...")
    plot_window(padded_sensor_data, padded_rul_data, sensor_names=sensor_names,
                mask=mask, title="Window Plot (With Padding)")

if __name__ == "__main__":
    main() 