# Hydroptimal


## Description

This GitHub repository supports the article "Comparing Time Series in Hydrogeology Using Optimal Transport" by Voland et al., which proposes a method for computing optimal transport (OT) distances to compare hydrogeological time series. The Python code "Optimal_transport.py" implements this method and allows for the quick computation of OT distances directly from the time series. Users must choose the hyperparameters reg and logeps (for Sinkhorn distances only) according to their needs. Preprocessing the time series depends on the values of k1 and k2. A higher difference between k1 and k2 results in an objective function that is more sensitive to mass differences than to time shifts.
For large time series (>1,000 time steps), speed-up techniques are necessary for computing Sinkhorn distances, as they require multiple matrix-vector multiplications, which can be time-consuming in Python if the matrix size is large.
This repository also contains three examples of the use of optimal transport distances in hydrological inverse problems: Gaussians, regular pulses, and tracer tests.

## Authors

Created by Robin Voland in 2026




