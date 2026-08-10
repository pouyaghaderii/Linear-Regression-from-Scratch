# Linear Regression from Scratch

## Why this project?

Most people learn linear regression by calling `sklearn.LinearRegression().fit()`
and never actually see what happens underneath. I wanted to build it myself —
derive the math by hand, implement it with only NumPy, and *prove* (not just
assume) that it actually works by checking my gradients numerically and
comparing my results against scikit-learn's implementation.

This is the first project in a series where I'm building up a machine
learning portfolio from the ground up — starting with the fundamentals
before moving on to applied and research-level work.

## What's implemented

- Linear regression trained with batch gradient descent, built from scratch
  with NumPy (no ML libraries used for the core algorithm)
- Numerical gradient checking, to verify the hand-derived gradient formulas
  are actually correct
- A comparison against scikit-learn's `LinearRegression` on the same data
- An exploratory notebook comparing different learning rates and their
  effect on convergence

## Problem setup

Given `m` training examples with `n` features, `X ∈ R^(m×n)`, and targets
`y ∈ R^m`, the goal is to learn a weight vector `w ∈ R^n` and a bias term
`b` such that the predictions `ŷ = Xw + b` are as close as possible to `y`,
measured by Mean Squared Error.

The full mathematical derivation (cost function, gradients, and how they're
implemented) is documented in `src/linear_regression.py` and explained
below as the project builds up.

## Status

🚧 Building this step by step — see commit history for the full process.

## Project structure

```
linear-regression-from-scratch/
├── src/            # core implementation
├── notebooks/       # exploratory experiments
├── results/          # generated plots
├── requirements.txt
└── LICENSE
```
