import torch
import numpy as np
from sklearn.metrics import mean_squared_error
from preprocessing import prepare_data, process_test_data
from plot_utils import load_model, plot_predictions, evaluate_predictions, compute_s_score

def get_last_predictions(predictions: np.ndarray, num_test_windows_list: list) -> np.ndarray:
    """
    Given the full predictions for all test windows and a list of number
    of test windows per engine, extracts the prediction corresponding to
    the last window for each engine.
    """
    # Split predictions into a list, one item per engine.
    preds_for_each_engine = np.split(predictions, np.cumsum(num_test_windows_list)[:-1])
    # For each engine, take the prediction from its last window.
    preds_for_last_example = np.array([preds[-1] for preds in preds_for_each_engine])
    return preds_for_last_example

if __name__ == "__main__":
    print("Evaluating models...")

    # Load and process test data.
    _, test_data, true_rul, window_length, shift, _ = prepare_data(
        train_path="../CMAPSSData/train_FD001.txt",
        test_path="../CMAPSSData/test_FD001.txt",
        rul_path="../CMAPSSData/RUL_FD001.txt"
    )
    
    # Process test data engine-by-engine.
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
    
    # Model configurations to evaluate.
    model_configs = [
        {
            "name": "TD Piecewise",
            "filepath": "saved_models/best_td_piecewise.pth",
            "plot": "results/rul_predictions_td_piecewise.png"
        },
        {
            "name": "TD Linear",
            "filepath": "saved_models/best_td_linear.pth",
            "plot": "results/rul_predictions_td_linear.png"
        },
        {
            "name": "MC Piecewise",
            "filepath": "saved_models/best_mc_piecewise.pth",
            "plot": "results/rul_predictions_mc_piecewise.png"
        },
        {
            "name": "MC Linear",
            "filepath": "saved_models/best_mc_linear.pth",
            "plot": "results/rul_predictions_mc_linear.png"
        }
    ]
    
    results = {}
    
    # Loop through each model configuration.
    for config in model_configs:
        print(f"\nEvaluating model: {config['name']}")
        # Load the model (assumes load_model sets the correct device).
        model = load_model(config["filepath"], input_channels=processed_test_data.shape[2])
        
        # Forward pass on test data.
        with torch.no_grad():
            test_tensor = torch.FloatTensor(processed_test_data).to(model.device)
            predictions = model(test_tensor).cpu().numpy()
        
        # Extract last prediction for each engine.
        preds_for_last_example = get_last_predictions(predictions, num_test_windows_list)
        
        # Compute metrics
        rmse = np.sqrt(mean_squared_error(true_rul, preds_for_last_example))
        s_score = compute_s_score(true_rul, preds_for_last_example)
        results[config["name"]] = {"RMSE": rmse, "S-score": s_score}
        
        # Evaluate and plot the results for this model.
        print(f"Evaluation results for {config['name']}:")
        evaluate_predictions(true_rul, preds_for_last_example)
        print(f"RMSE: {rmse:.4f}, S-score: {s_score:.4f}")
        plot_predictions(true_rul, preds_for_last_example, save_path=config["plot"], method_name=config["name"])
    
    # Summary comparison of models.
    print("\n\nSummary Comparison of Models:")
    
    # Compare TD models.
    print("\nTD Models Comparison:")
    td_piecewise = results.get("TD Piecewise", {})
    td_linear = results.get("TD Linear", {})
    print(f"TD Piecewise - RMSE: {td_piecewise.get('RMSE', np.nan):.4f}, S-score: {td_piecewise.get('S-score', np.nan):.4f}")
    print(f"TD Linear   - RMSE: {td_linear.get('RMSE', np.nan):.4f}, S-score: {td_linear.get('S-score', np.nan):.4f}")
    
    # Compare MC models.
    print("\nMC Models Comparison:")
    mc_piecewise = results.get("MC Piecewise", {})
    mc_linear = results.get("MC Linear", {})
    print(f"MC Piecewise - RMSE: {mc_piecewise.get('RMSE', np.nan):.4f}, S-score: {mc_piecewise.get('S-score', np.nan):.4f}")
    print(f"MC Linear   - RMSE: {mc_linear.get('RMSE', np.nan):.4f}, S-score: {mc_linear.get('S-score', np.nan):.4f}") 