# TD Moving Targte Problem

This project modifies the existing code for Continuous Monte Carlo (MC), Temporal Difference (TD) learning with TD(0), and TD(λ) algorithms.

## Overview

In Temporal Difference (TD) learning, each update uses the current estimate of the next state’s value $V(s')$ to form the target://
$V(s) \leftarrow V(s)+\alpha[r+\gamma V(s')-V(s)]$
- **Discrete_MC.py**: Defines the `Discrete_MC` class, which is used to create a discrete Markov chain with a specified number of states. It includes methods for generating the transition matrix with non-uniform behavior and for generating sample trajectories from the Markov chain.

- **methods.py**: Contains functions for estimating the MFPT using Monte Carlo simulation, TD(0), and TD(λ) from pre-generated samples of trajectories in the Markov chain.

- **data_construction.py**: Utilizes the `Discrete_MC` class and functions from `methods.py` to construct data and compare the mean-based approaches for estimating MFPT.

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
