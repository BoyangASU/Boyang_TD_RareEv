# TD Moving Target Problem

This project modifies the existing code for Continuous Monte Carlo (MC), Temporal Difference (TD) learning with TD(0), and TD(λ) algorithms.

## Overview

In Temporal Difference (TD) learning, each update uses the current estimate of the next state’s value $V(s')$ to form the target:
$V(s) \leftarrow V(s)+\alpha[r+\gamma V(s')-V(s)]$

However, because $V(s')$ is produced by the same model being updated, the target $r+\gamma V(s')$ keeps changing as learning progresses. This creates non-stationary targets, often causing:

- **Instability**: The model “chases” its own shifting predictions, potentially oscillating or diverging.

- **Slower Convergence**: Learning efficiency degrades due to rapidly moving objectives.

- **Inconsistent Estimates**: Value function updates can overestimate or underestimate, harming performance.
  
## Solutions: Double (Target) Networks
A double network (or target network) decouples the learning model from the target computation:

- **Main Network**: Actively trained on current data (transitions).

- **Target Network**: A lagged copy of the main network, updated periodically or slowly. This network alone computes $V(s')$.

Because the target network updates more gradually, the target $r+\gamma V_{target}(s')$ remains relatively stable, reducing non-stationarity and improving overall training stability.
 
## Requirements

- Python 3.x
- NumPy
- Matplotlib
- tqdm

## Usage

To use this project, here's an example of how to use `test.py` and compare Monte Carlo, TD(0), and TD(λ) methods:

``` python
python test.py
python evaluate.py
```

## Results 
![Comparison of Mean-Based Approaches](results/comparison_of_mean_based_approaches.png)

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## License

This project is open-source and available under the MIT License.
