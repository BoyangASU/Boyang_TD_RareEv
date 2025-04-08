import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Union, List, Optional

def process_targets(data_length: int, early_rul: int = None) -> np.ndarray:
    """Creates target RUL values based on data length and early RUL threshold."""
    if early_rul is None:
        return np.arange(data_length - 1, -1, -1)
    else:
        early_rul_duration = data_length - early_rul
        if early_rul_duration <= 0:
            return np.arange(data_length - 1, -1, -1)
        else:
            return np.append(early_rul * np.ones(shape=(early_rul_duration,)), 
                             np.arange(early_rul - 1, -1, -1))

def process_input_data_with_targets(input_data: np.ndarray, 
                                    target_data: np.ndarray = None,
                                    window_length: int = 1, 
                                    shift: int = 1,
                                    target_for_all_timestamps: bool = False,
                                    pad_value: float = 0
                                   ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Generate batches of data and targets using sliding windows.
    
    Parameters:
        input_data (np.ndarray): Data array of shape (num_timesteps, num_features).
        target_data (np.ndarray or None): 1D array of targets (one per timestep).
        window_length (int): The length of each sliding window.
        shift (int): Step size between windows.
        target_for_all_timestamps (bool): 
            If False, only the target corresponding to the last timestep in each window is returned.
            If True, a full window of targets is returned for every sliding window.
        pad_value (float): Value used to pad the sequence when its length is smaller than window_length.
    
    Returns:
        If target_data is None:
            np.ndarray of shape (num_batches, window_length, num_features)
        Else if target_for_all_timestamps is False:
            Tuple[np.ndarray, np.ndarray]:
                - data windows: shape (num_batches, window_length, num_features)
                - targets: shape (num_batches,) with each target taken at
                  index (shift*batch + (window_length-1))
        Else (target_for_all_timestamps is True):
            Tuple[np.ndarray, np.ndarray]:
                - data windows: shape (num_batches, window_length, num_features)
                - targets: shape (num_batches, window_length) containing the target
                  trajectory over the window.
    """
    num_timesteps = len(input_data)
    num_features = input_data.shape[1]
    
    # Automatically pad if input_data is shorter than window_length.
    if num_timesteps < window_length:
        padded_window = np.full((window_length, num_features), pad_value, dtype=input_data.dtype)
        padded_window[:num_timesteps, :] = input_data
        if target_data is None:
            return padded_window[None, ...]
        else:
            if not target_for_all_timestamps:
                target_val = target_data[-1]
                return padded_window[None, ...], np.array([target_val])
            else:
                padded_target = np.full((window_length,), pad_value, dtype=target_data.dtype)
                padded_target[:num_timesteps] = target_data
                return padded_window[None, ...], padded_target[None, ...]
    
    # Otherwise, create sliding windows normally.
    num_batches = int(np.floor((num_timesteps - window_length) / shift)) + 1
    output_data = np.empty((num_batches, window_length, num_features), dtype=input_data.dtype)
    
    if target_data is None:
        for batch in range(num_batches):
            start = shift * batch
            end = start + window_length
            output_data[batch, :, :] = input_data[start:end, :]
        return output_data
    else:
        if not target_for_all_timestamps:
            output_targets = np.empty((num_batches,), dtype=target_data.dtype)
            for batch in range(num_batches):
                start = shift * batch
                end = start + window_length
                output_data[batch, :, :] = input_data[start:end, :]
                output_targets[batch] = target_data[end - 1]
            return output_data, output_targets
        else:
            output_targets = np.empty((num_batches, window_length), dtype=target_data.dtype)
            for batch in range(num_batches):
                start = shift * batch
                end = start + window_length
                output_data[batch, :, :] = input_data[start:end, :]
                output_targets[batch, :] = target_data[start:end]
            return output_data, output_targets

def process_test_data(test_data_for_an_engine: np.ndarray, 
                     window_length: int,
                     shift: int, 
                     num_test_windows: int = 1) -> Tuple[np.ndarray, int]:
    """Process test data for an engine."""
    max_num_test_batches = int(np.floor((len(test_data_for_an_engine) - window_length)/shift)) + 1
    if max_num_test_batches < num_test_windows:
        required_len = (max_num_test_batches - 1) * shift + window_length
        batched_test_data = process_input_data_with_targets(
            test_data_for_an_engine[-required_len:, :],
            target_data=None,
            window_length=window_length,
            shift=shift
        )
        return batched_test_data, max_num_test_batches
    else:
        required_len = (num_test_windows - 1) * shift + window_length
        batched_test_data = process_input_data_with_targets(
            test_data_for_an_engine[-required_len:, :],
            target_data=None,
            window_length=window_length,
            shift=shift
        )
        return batched_test_data, num_test_windows

def prepare_data(train_path, test_path, rul_path, window_length: int = 30, shift: int = 1, early_rul: int = 125):
    """
    Prepare training and test data.
    """
    # Read data
    train_data = pd.read_csv(train_path, sep=r"\s+", header=None)
    test_data = pd.read_csv(test_path, sep=r"\s+", header=None)
    true_rul = pd.read_csv(rul_path, sep=r"\s+", header=None)
    
    # Columns to drop (based on reference implementation)
    columns_to_be_dropped = [0,1,2,3,4,5,9,10,14,20,22,23]
    
    # Store first columns before scaling
    train_data_first_column = train_data[0]
    test_data_first_column = test_data[0]
    
    # Scale data
    scaler = MinMaxScaler(feature_range=(-1,1))
    train_data_scaled = scaler.fit_transform(train_data.drop(columns=columns_to_be_dropped))
    test_data_scaled = scaler.transform(test_data.drop(columns=columns_to_be_dropped))
    
    # Reconstruct DataFrames with first column
    train_data = pd.DataFrame(data=np.c_[train_data_first_column, train_data_scaled])
    test_data = pd.DataFrame(data=np.c_[test_data_first_column, test_data_scaled])
    
    return train_data, test_data, true_rul[0].values, window_length, shift, early_rul 

def process_full_sequences_with_padding(sequences: List[np.ndarray], 
                                        target_sequences: Optional[List[np.ndarray]] = None,
                                        pad_value: float = 0) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Pads a list of sequences (and corresponding target sequences, if provided) to the maximum
    sequence length found across the batch and creates a binary mask.
    
    Parameters:
        sequences: List of np.ndarray, each of shape (seq_length, num_features)
        target_sequences: Optional list of np.ndarray, each of shape (seq_length, ) corresponding to targets.
        pad_value: Value to use for padding.
    
    Returns:
        padded_sequences: np.ndarray of shape (batch_size, max_seq_length, num_features)
        padded_targets: np.ndarray of shape (batch_size, max_seq_length) if target_sequences is provided, else None.
        mask: np.ndarray of shape (batch_size, max_seq_length) with 1 indicating valid timesteps, 0 for padded.
    """
    batch_size = len(sequences)
    max_seq_length = max(seq.shape[0] for seq in sequences)
    num_features = sequences[0].shape[1] if batch_size > 0 else 0
    
    padded_sequences = np.full((batch_size, max_seq_length, num_features), pad_value, dtype=sequences[0].dtype)
    mask = np.zeros((batch_size, max_seq_length), dtype=np.uint8)
    
    padded_targets = None
    if target_sequences is not None:
        # Assuming targets are 1D arrays.
        padded_targets = np.full((batch_size, max_seq_length), pad_value, dtype=target_sequences[0].dtype)
    
    for i, seq in enumerate(sequences):
        seq_length = seq.shape[0]
        padded_sequences[i, :seq_length, :] = seq
        mask[i, :seq_length] = 1
        if target_sequences is not None:
            padded_targets[i, :seq_length] = target_sequences[i]
    
    return padded_sequences, padded_targets, mask 

def sort_samples_by_rul(data: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a set of samples and their corresponding target RUL values,
    return the samples sorted in ascending order based on the RUL,
    so that terminal transitions (with lower RUL) appear first.

    Parameters:
        data (np.ndarray): Array of training samples, e.g., shape (num_samples, window_length, num_features)
        targets (np.ndarray): Array of target RUL values with shape (num_samples,)

    Returns:
        sorted_data (np.ndarray): Data sorted according to ascending RUL.
        sorted_targets (np.ndarray): Targets sorted in ascending order.
    """
    indices = np.argsort(targets)  # Ascending order: lower (terminal) RUL first.
    return data[indices], targets[indices] 