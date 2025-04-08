import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

def create_terminal_oversampled_loader(dataset, terminal_threshold: float = 0.0, 
                                         oversample_multiplier: float = 5.0, 
                                         batch_size: int = 32) -> DataLoader:
    """
    Creates a DataLoader that oversamples transitions near or at terminal states.
    
    Parameters:
      dataset: A PyTorch Dataset returning (data, target) pairs; target is expected to be a tensor scalar
               or a 1D tensor where the last element is used as the terminal indicator.
      terminal_threshold: Samples with target values <= this threshold are considered terminal.
      oversample_multiplier: The factor by which terminal samples are oversampled.
      batch_size: The batch size for the DataLoader.
    
    Returns:
      A DataLoader that uses a WeightedRandomSampler to increase the occurrence
      of terminal transitions in each batch.
    """
    # Compute sample weights.
    weights = []
    for i in range(len(dataset)):
        _, target = dataset[i]
        # Ensure target is a float value.
        if isinstance(target, torch.Tensor):
            if target.dim() > 0:
                # Use the last element if target is a vector.
                target_value = float(target[-1].item())
            else:
                target_value = float(target.item())
        else:
            target_value = float(target)
        
        if target_value <= terminal_threshold:
            weights.append(oversample_multiplier)
        else:
            weights.append(1.0)
    
    sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    
    return loader

# Example usage
if __name__ == "__main__":
    from train import RULDataset  # Assuming your dataset is defined in train.py
    import numpy as np
    
    # Dummy data to illustrate. Replace this with your actual preprocessed training data.
    dummy_data = np.random.randn(1000, 30, 14).astype("float32")  # (num_samples, window_length, num_features)
    # Suppose targets are generated such that a few are terminal (0) and the rest are higher.
    dummy_targets = np.random.uniform(50, 125, size=(1000,)).astype("float32")
    # Manually force some terminal transitions:
    dummy_targets[:50] = 0.0  # first 50 are terminal
    
    # Create the dataset.
    dataset = RULDataset(dummy_data, dummy_targets)
    
    # Create an oversampled DataLoader.
    loader = create_terminal_oversampled_loader(dataset, terminal_threshold=0.0, 
                                                oversample_multiplier=10.0, batch_size=16)
    
    # Check the distribution in one epoch:
    terminal_count = 0
    total_count = 0
    for _, targets in loader:
        total_count += targets.size(0)
        terminal_count += (targets <= 0.0).sum().item()
    
    print(f"In one epoch, sampled {terminal_count} terminal transitions out of {total_count} total samples.") 