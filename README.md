# CSE5403 Project: Hyperparameter Optimization for FashionMNIST Classification

This project implements hyperparameter optimization (HPO) for convolutional neural networks (CNNs) and Transformer models on the FashionMNIST dataset using PyTorch and Optuna.

## Project Overview

The project explores various HPO algorithms including:
- Bayesian Optimization (BO)
- Asynchronous Successive Halving Algorithm (ASHA)
- Budgeted Neural Architecture Search (BNS)
- Bayesian Optimization with Hyperband (BOHB)
- Multi-Fidelity Bayesian Optimization (MFBO)
- Trust Region Bayesian Optimization

## Files Description

- `combined.ipynb`: Jupyter notebook containing the complete project workflow, including data loading, model training, HPO experiments, and result analysis.
- `core_model.py`: Core model definitions and training functions for CNN and Transformer architectures.
- `run_hpo.py`: Script to run hyperparameter optimization using different algorithms.
- `analyze_best_acc.py`: Script to analyze and compare the best accuracies achieved by different HPO methods.
- `plot_results.py`: Visualization scripts for plotting HPO results and comparisons.
- `test.py`: Test scripts for model evaluation.
- `bo_logs/`: Directory containing logs from various HPO trials in JSONL format.
- `data/FashionMNIST/`: FashionMNIST dataset files.
- `pics/` and `new_pics/`: Directories for storing generated plots and images.

## Dependencies

- Python 3.8+
- PyTorch
- Optuna
- NumPy
- Matplotlib
- Seaborn
- Pandas
- Scikit-learn

Install dependencies using:
```bash
pip install torch torchvision optuna numpy matplotlib seaborn pandas scikit-learn
```

## How to Run

1. **Data Preparation**: The FashionMNIST dataset will be automatically downloaded when running the notebook or scripts.

2. **Run HPO Experiments**:
   ```bash
   python run_hpo.py --model cnn --algorithm bo --n_trials 50
   ```

3. **Analyze Results**:
   ```bash
   python analyze_best_acc.py
   ```

4. **Plot Results**:
   ```bash
   python plot_results.py
   ```

5. **Run the Complete Notebook**: Open `combined.ipynb` and execute all cells, this is code for grid search, random search and standard BO.


## Grid Search

Grid Search exhaustively evaluates combinations of hyperparameters from predefined candidate sets. For this project, only 30 out of 240 combination were tested due to computational cost. Each configuration trains a CNN (Model 2) for 3 epochs on the training set, and the best validation accuracy from that run is recorded. All grid search trials are logged in JSONL format under: bo_logs/grid_trials.jsonl.

## Random Search

Random Search samples hyperparameters uniformly at random from predefined candidate sets. All trials are logged in JSONL format under: bo_logs/random_trials.jsonl.

## Bayesian Optimization (Optuna)

Bayesian Optimization is implemented using Optuna to tune a CNN baseline on FashionMNIST. The model is trained with an 8:2 train/validation split. For each trial, Optuna searches over hidden_units, batch_size, learning rate, and weight_decay. Each configuration is trained for 3 epochs, and the best validation accuracy within the trial is recorded.

All trials are logged in JSONL format under `bo_logs/bo_trials.jsonl`, including hyperparameters, best validation accuracy, training time, and per-epoch metrics. These logs are then parsed to generate plots showing validation accuracy versus trial index, cumulative training time, and individual trial performance.



## Results

The project compares different HPO algorithms on CNN and Transformer models, evaluating their performance in terms of accuracy and computational efficiency.

## Contributors

- Zheng Fang
- Ingrid Jin
- Mian Dai
- Ruifu (Jeff) Chen

## License

This project is for educational purposes as part of CSE5403 course.
