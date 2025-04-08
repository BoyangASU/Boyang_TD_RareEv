#%% Downloaded from https://github.com/biswajitsahoo1111/rul_codes_open/blob/master/notebooks/cmapss_notebooks/CMAPSS_FD001_1D_CNN_piecewise_linear_degradation_model.ipynb

from preprocessing import prepare_data, process_test_data, process_input_data_with_targets, process_targets
from train import train_model
from plot_utils import plot_predictions, evaluate_predictions, load_model, compute_s_score
import torch
import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def train_and_evaluate(train_data, test_data, true_rul, window_length, shift, early_rul=None, model_name="model"):
    """
    Train and evaluate a RUL model.
    
    For each engine:
      - Process training data: slide the window over the sensor data and generate targets.
      - Process test data similarly.
    
    The processed data is then used by train_model; predictions for the last window of each engine are returned.
    """
    processed_train_data = []
    processed_train_targets = []
    processed_test_data = []
    num_test_windows_list = []
    
    # Process training data per engine.
    num_train_machines = len(train_data[0].unique())
    for i in range(1, num_train_machines + 1):
        # Get sensor values only.
        temp_train_data = train_data[train_data[0] == i].drop(columns=[0]).values
        temp_train_targets = process_targets(data_length=temp_train_data.shape[0], early_rul=early_rul)
        
        # Create sliding windows for data and targets.
        data_for_machine, targets_for_machine = process_input_data_with_targets(
            temp_train_data, temp_train_targets, window_length=window_length, shift=shift
        )
        processed_train_data.append(data_for_machine)
        processed_train_targets.append(targets_for_machine)
    
    processed_train_data = np.concatenate(processed_train_data)
    processed_train_targets = np.concatenate(processed_train_targets)
    
    # Process test data per engine.
    num_test_machines = len(test_data[0].unique())
    for i in range(1, num_test_machines + 1):
        temp_test_data = test_data[test_data[0] == i].drop(columns=[0]).values
        # Process test data; if there is not enough data the preprocessing function will raise an error.
        test_data_for_engine, num_windows = process_test_data(
            temp_test_data, window_length=window_length, shift=shift, num_test_windows=1
        )
        processed_test_data.append(test_data_for_engine)
        num_test_windows_list.append(num_windows)
    
    processed_test_data = np.concatenate(processed_test_data)
    
    # Train the model.
    model = train_model(
        processed_train_data, 
        processed_train_targets,
        model_save_path="saved_models",
        model_name="mc_" + model_name
    )
    
    # Generate predictions.
    model.eval()
    with torch.no_grad():
        test_tensor = torch.FloatTensor(processed_test_data).to(model.device)
        predictions = model(test_tensor).cpu().numpy()
    
    # Reconstruct predictions per engine.
    preds_for_each_engine = np.split(predictions, np.cumsum(num_test_windows_list)[:-1])
    indices_of_last_examples = np.cumsum(num_test_windows_list) - 1
    preds_for_last_example = np.concatenate(preds_for_each_engine)[indices_of_last_examples]
    
    return true_rul, preds_for_last_example

#%%
if __name__ == "__main__":
    # Create directories
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Prepare data
    train_data, test_data, true_rul, window_length, shift, _ = prepare_data(
        train_path="../CMAPSSData/train_FD001.txt",
        test_path="../CMAPSSData/test_FD001.txt",
        rul_path="../CMAPSSData/RUL_FD001.txt"
    )
    
    # Train and evaluate piecewise model (early_rul=125)
    print("\nTraining Piecewise RUL Model...")
    true_rul_pw, pred_rul_pw = train_and_evaluate(
        train_data, test_data, true_rul, window_length, shift, 
        early_rul=125, model_name="piecewise"
    )
    
    # Train and evaluate linear model
    print("\nTraining Linear RUL Model...")
    true_rul_lin, pred_rul_lin = train_and_evaluate(
        train_data, test_data, true_rul, window_length, shift, 
        early_rul=None, model_name="linear"
    )       
    
    # Compare results
    print("\nPiecewise RUL Model Results:")
    rmse_pw = np.sqrt(mean_squared_error(true_rul_pw, pred_rul_pw))
    s_score_pw = compute_s_score(true_rul_pw, pred_rul_pw)
    print(f"RMSE: {rmse_pw:.4f}")
    print(f"S-score: {s_score_pw:.4f}")
    
    print("\nLinear RUL Model Results:")
    rmse_lin = np.sqrt(mean_squared_error(true_rul_lin, pred_rul_lin))
    s_score_lin = compute_s_score(true_rul_lin, pred_rul_lin)
    print(f"RMSE: {rmse_lin:.4f}")
    print(f"S-score: {s_score_lin:.4f}")
    
    # Plot results
    plot_predictions(true_rul_pw, pred_rul_pw, save_path="results/rul_predictions_piecewise.png")
    plot_predictions(true_rul_lin, pred_rul_lin, save_path="results/rul_predictions_linear.png")
