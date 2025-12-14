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

5. **Run the Complete Notebook**: Open `combined.ipynb` and execute all cells for a comprehensive walkthrough.

## Results

The project compares different HPO algorithms on CNN and Transformer models, evaluating their performance in terms of accuracy and computational efficiency.

## Contributors

- [Your Name]
- [Team Member 1]
- [Team Member 2]
- ...

## License

This project is for educational purposes as part of CSE5403 course.