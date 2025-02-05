# TD Moving Targte Problem

This project modifies the existing code for Continuous Monte Carlo (MC), Temporal Difference (TD) learning with TD(0), and TD(λ) algorithms.

## Overview

The project is structured into three main components:

- **Discrete_MC.py**: Defines the `Discrete_MC` class, which is used to create a discrete Markov chain with a specified number of states. It includes methods for generating the transition matrix with non-uniform behavior and for generating sample trajectories from the Markov chain.

- **methods.py**: Contains functions for estimating the MFPT using Monte Carlo simulation, TD(0), and TD(λ) from pre-generated samples of trajectories in the Markov chain.

- **data_construction.py**: Utilizes the `Discrete_MC` class and functions from `methods.py` to construct data and compare the mean-based approaches for estimating MFPT.

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- tqdm

## Usage

To use this project, follow these steps:

1. Ensure all requirements are installed by running:
   ```bash
   pip install numpy matplotlib tqdm
   ```

2. Generate the Markov chain and sample trajectories using `Discrete_MC.py`. This can be done by creating an instance of the `Discrete_MC` class and calling its `generate_samples` method.

3. Estimate the MFPT using the provided methods in `methods.py`. This involves calling the `monte_carlo_estimator_from_samples`, `td_estimator_from_samples`, or `td_lambda_estimator_from_samples` functions with the generated samples as input.

4. Compare the different mean-based approaches by running `data_construction.py`, which will plot the results of the estimations for visual comparison.

## Example
### Generate Data 
``` python
from Discrete_MC import Discrete_MC
# Initialize the Markov Chain
n = 20 # Number of states
mc_simulation = Discrete_MC(n)
#Generate samples
num_samples = 500
samples = mc_simulation.generate_samples(num_samples)
```
This example demonstrates how to initialize the Markov chain with 20 states and generate 500 sample trajectories.

For more detailed examples, including how to run the estimations and compare the results, refer to the `data_construction.py` script.

### Generate Data 
### Compute Mean First Passage Time

Here's an example of how to use `test.py` and compare Monte Carlo, TD(0), and TD(λ) methods:

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
