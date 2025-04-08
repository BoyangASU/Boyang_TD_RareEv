import os
import numpy as np
import unittest
import matplotlib.pyplot as plt
from preprocessing import prepare_data, process_input_data_with_targets, process_targets, process_full_sequences_with_padding
from plot_utils import plot_window  # Universal plotting function

# Global variables to control debug output and window length
DEBUG_PRINT = False
CURRENT_WINDOW_LENGTH = 30

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        # Use the global window length
        self.window_length = CURRENT_WINDOW_LENGTH
        if DEBUG_PRINT:
            print(f"[DEBUG] Setting up test with window_length = {self.window_length}")

        # Paths to the data files
        train_path = os.path.join("..", "CMAPSSData", "train_FD001.txt")
        test_path = os.path.join("..", "CMAPSSData", "test_FD001.txt")
        rul_path = os.path.join("..", "CMAPSSData", "RUL_FD001.txt")
        
        # Prepare data (window_length is passed here)
        self.train_data, self.test_data, self.true_rul, _, self.shift, self.early_rul = prepare_data(
            train_path=train_path,
            test_path=test_path,
            rul_path=rul_path,
            window_length=self.window_length  # Pass current window length
        )
        
        # Select one engine's data (engine id == 1) from train_data.
        engine1_data = self.train_data[self.train_data[0] == 1].drop(columns=[0])
        self.input_data = engine1_data.values  # Sensor data with 14 features
        self.num_timesteps = self.input_data.shape[0]
        self.num_features = self.input_data.shape[1]
        
        # Process targets
        self.target_data = process_targets(data_length=self.num_timesteps, early_rul=self.early_rul)
def test_shape_verification_single_target(self):
    processed_data, processed_targets = process_input_data_with_targets(
        self.input_data,
        target_data=self.target_data,
        window_length=self.window_length,
        shift=self.shift,
        target_for_all_timestamps=False
    )
    # If the available timesteps are less than the window_length, expect 1 batch (padded)
    if self.num_timesteps < self.window_length:
        expected_num_batches = 1
        # Expected padded window: first num_timesteps rows are original; remaining rows are pad (0).
        expected_window = np.full((self.window_length, self.num_features), 0, dtype=self.input_data.dtype)
        expected_window[:self.num_timesteps, :] = self.input_data
        expected_target = self.target_data[-1]  # Single target: last value.
    else:
        expected_num_batches = int(np.floor((self.num_timesteps - self.window_length) / self.shift)) + 1

    if DEBUG_PRINT:
        print(f"[DEBUG] Single target mode:")
        print(f"  Input shape: {self.input_data.shape}")
        print(f"  Processed data shape: {processed_data.shape}")
        print(f"  Processed targets shape: {processed_targets.shape}")
        print(f"  Expected number of batches: {expected_num_batches}")
    self.assertEqual(processed_data.shape, (expected_num_batches, self.window_length, self.num_features))
    self.assertEqual(processed_targets.shape, (expected_num_batches,))
    if self.num_timesteps < self.window_length:
        np.testing.assert_array_equal(processed_data[0], expected_window)
        self.assertEqual(processed_targets[0], expected_target)


def test_shape_verification_full_targets(self):
    processed_data, processed_targets = process_input_data_with_targets(
        self.input_data,
        target_data=self.target_data,
        window_length=self.window_length,
        shift=self.shift,
        target_for_all_timestamps=True
    )
    # For full-target mode, if the input is too short, we expect one padded window.
    if self.num_timesteps < self.window_length:
        expected_num_batches = 1
        expected_window = np.full((self.window_length, self.num_features), 0, dtype=self.input_data.dtype)
        expected_window[:self.num_timesteps, :] = self.input_data
        expected_target = np.full((self.window_length,), 0, dtype=self.target_data.dtype)
        expected_target[:self.num_timesteps] = self.target_data
    else:
        expected_num_batches = int(np.floor((self.num_timesteps - self.window_length) / self.shift)) + 1

    if DEBUG_PRINT:
        print(f"[DEBUG] Full targets mode:")
        print(f"  Input shape: {self.input_data.shape}")
        print(f"  Processed data shape: {processed_data.shape}")
        print(f"  Processed targets shape: {processed_targets.shape}")
        if expected_num_batches == 1:
            print(f"  Expected padded targets: {expected_target}")
    self.assertEqual(processed_data.shape, (expected_num_batches, self.window_length, self.num_features))
    self.assertEqual(processed_targets.shape, (expected_num_batches, self.window_length))
    if self.num_timesteps < self.window_length:
        np.testing.assert_array_equal(processed_data[0], expected_window)
        np.testing.assert_array_equal(processed_targets[0], expected_target)


def test_original_data_slicing(self):
    processed_data, _ = process_input_data_with_targets(
        self.input_data,
        target_data=self.target_data,
        window_length=self.window_length,
        shift=self.shift,
        target_for_all_timestamps=True
    )
    # If input is shorter than window_length, expect one padded window.
    if self.num_timesteps < self.window_length:
        expected_num_batches = 1
        expected_window = np.full((self.window_length, self.num_features), 0, dtype=self.input_data.dtype)
        expected_window[:self.num_timesteps, :] = self.input_data
    else:
        expected_num_batches = int(np.floor((self.num_timesteps - self.window_length) / self.shift)) + 1
        # For this test, we are checking specific windows.
        idx = 0  # Checking the first window.
        start = idx * self.shift
        end = start + self.window_length
        expected_window = self.input_data[start:end, :]

    if DEBUG_PRINT:
        print(f"[DEBUG] Testing original data slicing:")
        print(f"  Processed data shape: {processed_data.shape}")
        print(f"  Expected window shape: {expected_window.shape}")
    self.assertEqual(processed_data.shape[0], expected_num_batches)
    np.testing.assert_array_equal(processed_data[0], expected_window)

    def test_last_timestamp_matches(self):
        processed_data, processed_targets = process_input_data_with_targets(
            self.input_data,
            target_data=self.target_data,
            window_length=self.window_length,
            shift=self.shift,
            target_for_all_timestamps=False
        )
        expected_num_batches = int(np.floor((self.num_timesteps - self.window_length) / self.shift)) + 1
        for batch in range(expected_num_batches):
            sliced_targets = self.target_data[(self.shift * batch):(self.shift * batch + self.window_length)]
            expected_value = sliced_targets[-1]
            if DEBUG_PRINT:
                print(f"[DEBUG] Window {batch}: Expected target {expected_value} from slice {sliced_targets}")
            self.assertEqual(processed_targets[batch], expected_value)

    def test_original_data_slicing(self):
        processed_data, _ = process_input_data_with_targets(
            self.input_data,
            target_data=self.target_data,
            window_length=self.window_length,
            shift=self.shift,
            target_for_all_timestamps=True
        )
        expected_num_batches = int(np.floor((self.num_timesteps - self.window_length) / self.shift)) + 1
        # Test first, middle, and last windows.
        for idx in [0, expected_num_batches // 2, expected_num_batches - 1]:
            start = idx * self.shift
            end = start + self.window_length
            expected_window = self.input_data[start:end, :]
            if DEBUG_PRINT:
                print(f"[DEBUG] Testing window index {idx}: Expected slice from {start} to {end}.")
            np.testing.assert_array_equal(processed_data[idx], expected_window)

    # NEW TESTS for process_full_sequences_with_padding
    def test_process_full_sequences_with_padding_without_targets(self):
        # Create a list of sequences with variable lengths.
        seq1 = np.array([[1, 2], [3, 4], [5, 6]])         # shape: (3, 2)
        seq2 = np.array([[7, 8], [9, 10]])                  # shape: (2, 2)
        seq3 = np.array([[11, 12], [13, 14], [15, 16], [17, 18]])  # shape: (4, 2)
        sequences = [seq1, seq2, seq3]
        
        padded_seqs, padded_targets, mask = process_full_sequences_with_padding(sequences, target_sequences=None, pad_value=0)
        
        # Expected max length = 4, and batch size = 3.
        self.assertEqual(padded_seqs.shape, (3, 4, 2))
        # padded_targets should be None.
        self.assertIsNone(padded_targets)
        # Expected mask: first row: [1,1,1,0], second row: [1,1,0,0], third row: [1,1,1,1]
        expected_mask = np.array([[1, 1, 1, 0],
                                  [1, 1, 0, 0],
                                  [1, 1, 1, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(mask, expected_mask)

    def test_process_full_sequences_with_padding_with_targets(self):
        # Create a list of sequences with variable lengths.
        seq1 = np.array([[1, 2], [3, 4]])         # shape: (2, 2)
        seq2 = np.array([[5, 6], [7, 8], [9, 10]])  # shape: (3, 2)
        sequences = [seq1, seq2]
        # Create corresponding target sequences.
        targets1 = np.array([10, 11])
        targets2 = np.array([12, 13, 14])
        target_seqs = [targets1, targets2]
        
        padded_seqs, padded_targets, mask = process_full_sequences_with_padding(sequences, target_sequences=target_seqs, pad_value=-1)
        
        # Expected max length = 3, and batch size = 2.
        self.assertEqual(padded_seqs.shape, (2, 3, 2))
        self.assertEqual(padded_targets.shape, (2, 3))
        
        # For seq1, padded row should be the original values and then the pad value -1.
        expected_seq1 = np.array([[1, 2], [3, 4], [-1, -1]])
        # For seq2, should be unchanged.
        expected_seq2 = np.array([[5, 6], [7, 8], [9, 10]])
        expected_padded = np.array([expected_seq1, expected_seq2])
        np.testing.assert_array_equal(padded_seqs, expected_padded)
        
        # Expected targets: for seq1: [10, 11, -1], for seq2: [12, 13, 14]
        expected_targets = np.array([[10, 11, -1],
                                     [12, 13, 14]])
        np.testing.assert_array_equal(padded_targets, expected_targets)
        
        # Expected mask: seq1: [1,1,0], seq2: [1,1,1]
        expected_mask = np.array([[1, 1, 0],
                                  [1, 1, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(mask, expected_mask)

def visualize_padded_real_data():
    """
    Load real data from the CMAPSS train set, group the data by engine id,
    and then pad the sequences using process_full_sequences_with_padding.
    Finally, visualize the padded result of one engine using the universal plot_window function.
    """
    from preprocessing import prepare_data, process_full_sequences_with_padding, process_targets
    from plot_utils import plot_window
    import os

    # Paths to the real data files.
    train_path = os.path.join("..", "CMAPSSData", "train_FD001.txt")
    test_path = os.path.join("..", "CMAPSSData", "test_FD001.txt")
    rul_path = os.path.join("..", "CMAPSSData", "RUL_FD001.txt")
    
    # Prepare the data.
    train_data, _, _, window_length, _, early_rul = prepare_data(train_path, test_path, rul_path)
    
    # Get the unique engine IDs (assumed to be in the first column).
    engine_ids = train_data[0].unique()
    
    sequences = []
    target_sequences = []
    # For demonstration, we take the first three engines.
    for engine in engine_ids[:3]:
        engine_df = train_data[train_data[0] == engine].drop(columns=[0])
        seq = engine_df.values  # Sensor data for this engine.
        sequences.append(seq)
        # Compute the target sequence for the engine.
        targ = process_targets(data_length=seq.shape[0], early_rul=early_rul)
        target_sequences.append(targ)
    
    # Pad the sequences and targets.
    padded_seqs, padded_targets, mask = process_full_sequences_with_padding(
        sequences, target_sequences=target_sequences, pad_value=-1
    )
    
    # For visualization, select the first engine.
    sensor_names = [f"Sensor {i+1}" for i in range(padded_seqs.shape[2])]
    plot_window(
        window_data=padded_seqs[0],
        window_rul=padded_targets[0],
        sensor_names=sensor_names,
        mask=mask[0],
        title=f"Real Data Padded Window for Engine {engine_ids[0]}"
    )

def run_tests_and_print_sample(window_length=30, print_debug=False):
    """
    Run tests and display sample processed data with a specified window length.
    """
    global CURRENT_WINDOW_LENGTH, DEBUG_PRINT
    DEBUG_PRINT = print_debug
    CURRENT_WINDOW_LENGTH = window_length  # Update global window length

    # Create and run the test suite.
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPreprocessing)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful() and print_debug:
        print(f"\nAll tests passed for window length {window_length}! Displaying sample processed data.")
        # Create a test instance and call setUp manually to build the data.
        test_instance = TestPreprocessing()
        test_instance.setUp()
        processed_data, processed_targets = process_input_data_with_targets(
            test_instance.input_data,
            target_data=test_instance.target_data,
            window_length=test_instance.window_length,
            shift=test_instance.shift,
            target_for_all_timestamps=True
        )
        
        print(f"[DEBUG] Using window_length: {test_instance.window_length}")
        print("[DEBUG] Processed data shape:", processed_data.shape)
        print("[DEBUG] Processed targets shape:", processed_targets.shape)
        print("\n[DEBUG] Sample data for the first window (features):")
        print(processed_data[0])
        print("\n[DEBUG] Sample target values for the first window:")
        print(processed_targets[0])
        
        # Use the universal plotting function from plot_utils to plot the first window.
        from plot_utils import plot_window  # Import the universal plot_window function.
        
        # Define sensor names based on the number of features.
        sensor_names = [f"Sensor {i+1}" for i in range(test_instance.num_features)]
        
        # Call plot_window with the first window of sensor data and corresponding target (RUL) values.
        plot_window(
            window_data=processed_data[0],
            window_rul=processed_targets[0],
            sensor_names=sensor_names,
            title=f"Sample Window Plot (Window Length: {test_instance.window_length})"
        )
        
    elif not result.wasSuccessful():
        print("\nSome tests failed. Sample data will not be printed.")

if __name__ == '__main__':
    # Example usage: run tests with different window lengths.
    window_sizes = [50, 100, 200]
    for window_size in window_sizes:
        print(f"\nTesting with window size: {window_size}")
        try:
            run_tests_and_print_sample(window_length=window_size, print_debug=True)
        except ValueError as e:
            print(f"[ERROR] {e}")