# run_hpo.py
import argparse
import optuna
import numpy as np
from typing import Dict, Any, List

from core_model import train_once, LOG_DIR

# 搜索空间
SEARCH_SPACE_CNN = {
    "hidden_units": [16, 32, 64, 128],
    "batch_size": [32, 64, 128],
    "lr": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    "weight_decay": [0.0, 1e-5, 1e-4, 1e-3],
}

SEARCH_SPACE_TRANSFORMER = {
    "embed_dim": [64, 128],
    "num_heads": [4, 8],
    "num_layers": [1, 2],
    "batch_size": [32, 64],
    "lr": [1e-4, 3e-4, 1e-3],
    "weight_decay": [0.0, 1e-5, 1e-4],
}


def sample_params_from_trial(trial: optuna.trial.Trial, model_type: str = "cnn") -> Dict[str, Any]:
    if model_type == "cnn":
        hidden_units = trial.suggest_categorical("hidden_units", SEARCH_SPACE_CNN["hidden_units"])
        batch_size = trial.suggest_categorical("batch_size", SEARCH_SPACE_CNN["batch_size"])
        lr = trial.suggest_categorical("lr", SEARCH_SPACE_CNN["lr"])
        weight_decay = trial.suggest_categorical("weight_decay", SEARCH_SPACE_CNN["weight_decay"])
        return {
            "hidden_units": hidden_units,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
        }
    elif model_type == "transformer":
        embed_dim = trial.suggest_categorical("embed_dim", SEARCH_SPACE_TRANSFORMER["embed_dim"])
        num_heads = trial.suggest_categorical("num_heads", SEARCH_SPACE_TRANSFORMER["num_heads"])
        num_layers = trial.suggest_categorical("num_layers", SEARCH_SPACE_TRANSFORMER["num_layers"])
        batch_size = trial.suggest_categorical("batch_size", SEARCH_SPACE_TRANSFORMER["batch_size"])
        lr = trial.suggest_categorical("lr", SEARCH_SPACE_TRANSFORMER["lr"])
        weight_decay = trial.suggest_categorical("weight_decay", SEARCH_SPACE_TRANSFORMER["weight_decay"])
        return {
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
        }
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# =============== 1. BO (TPE) ===============
def run_bo(n_trials: int, model_type: str = "cnn"):
    def objective(trial: optuna.trial.Trial) -> float:
        params = sample_params_from_trial(trial, model_type)
        best_val_acc, total_time, _ = train_once(
            params,
            num_epochs=3,
            log_file=f"{model_type}_bo_trials.jsonl",
            trial=trial,
            enable_pruning=False,
            method_name=f"{model_type}_bo",
            trial_index=trial.number,
            compute_test_metrics=True,
            model_type=model_type,
        )
        return best_val_acc

    study = optuna.create_study(
        study_name=f"{model_type}_bo_fashionmnist_model",
        direction="maximize",
        sampler=optuna.samplers.TPESampler()
    )
    study.optimize(objective, n_trials=n_trials)
    return study


# =============== 2. BOHB (TPE + HyperbandPruner) ===============
def run_bohb(n_trials: int, max_epochs: int = 12, model_type: str = "cnn"):
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=1,
        max_resource=max_epochs,
        reduction_factor=3
    )

    def objective(trial: optuna.trial.Trial) -> float:
        params = sample_params_from_trial(trial, model_type)
        best_val_acc, total_time, _ = train_once(
            params,
            num_epochs=max_epochs,
            log_file=f"{model_type}_bohb_trials.jsonl",
            trial=trial,
            enable_pruning=True,
            method_name=f"{model_type}_bohb",
            trial_index=trial.number,
            compute_test_metrics=True,
            model_type=model_type,
        )
        return best_val_acc

    study = optuna.create_study(
        study_name=f"{model_type}_bohb_fashionmnist_model",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        pruner=pruner,
    )
    study.optimize(objective, n_trials=n_trials)
    return study


# =============== 3. MFBO ===============
def run_mfbo(n_trials: int, max_epochs: int = 12, model_type: str = "cnn"):
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=1,
        max_resource=max_epochs,
        reduction_factor=3
    )

    def objective(trial: optuna.trial.Trial) -> float:
        params = sample_params_from_trial(trial, model_type)
        if trial.number < n_trials // 2:
            num_epochs = 3
        else:
            num_epochs = max_epochs

        best_val_acc, total_time, _ = train_once(
            params,
            num_epochs=num_epochs,
            log_file=f"{model_type}_mfbo_trials.jsonl",
            trial=trial,
            enable_pruning=True,
            method_name=f"{model_type}_mfbo",
            trial_index=trial.number,
            compute_test_metrics=True,
            model_type=model_type,
        )
        return best_val_acc

    study = optuna.create_study(
        study_name=f"{model_type}_mfbo_fashionmnist_model",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        pruner=pruner,
    )
    study.optimize(objective, n_trials=n_trials)
    return study


# =============== 4. ASHA / Successive Halving ===============
def _make_asha_pruner(max_epochs: int):

    if hasattr(optuna.pruners, "AsynchronousSuccessiveHalvingPruner"):
        return optuna.pruners.AsynchronousSuccessiveHalvingPruner(
            min_resource=1,
            max_resource=max_epochs,
            reduction_factor=3,
        )
    else:
        return optuna.pruners.SuccessiveHalvingPruner(
            min_resource=1,
            reduction_factor=3,
            min_early_stopping_rate=0,
        )



def run_asha(n_trials: int, max_epochs: int = 12, model_type: str = "cnn"):
    pruner = _make_asha_pruner(max_epochs)

    def objective(trial: optuna.trial.Trial) -> float:
        params = sample_params_from_trial(trial, model_type)
        best_val_acc, total_time, _ = train_once(
            params,
            num_epochs=max_epochs,
            log_file=f"{model_type}_asha_trials.jsonl",
            trial=trial,
            enable_pruning=True,
            method_name=f"{model_type}_asha",
            trial_index=trial.number,
            compute_test_metrics=True,
            model_type=model_type,
        )
        return best_val_acc

    study = optuna.create_study(
        study_name=f"{model_type}_asha_fashionmnist_model",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        pruner=pruner,
    )
    study.optimize(objective, n_trials=n_trials)
    return study


# =============== 5. Trust Region BO ===============
def run_trust_region(n_trials: int, model_type: str = "cnn"):
    if model_type == "cnn":
        search_space = SEARCH_SPACE_CNN
    elif model_type == "transformer":
        search_space = SEARCH_SPACE_TRANSFORMER
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    idx_space = {
        k: np.arange(len(v)) for k, v in search_space.items()
    }

    tr_range = {
        k: (0, len(v) - 1) for k, v in idx_space.items()
    }

    results: List[Dict[str, Any]] = []

    for t in range(n_trials):
        if t > 0 and t % 10 == 0:
            best = max(results, key=lambda r: r["value"])
            best_idx = best["idx_params"]
            radius = 1
            for k in tr_range.keys():
                low = max(0, best_idx[k] - radius)
                high = min(len(idx_space[k]) - 1, best_idx[k] + radius)
                tr_range[k] = (low, high)

        idx_params = {}
        params = {}
        for name, (low, high) in tr_range.items():
            idx_val = np.random.randint(low, high + 1)
            idx_params[name] = idx_val
            params[name] = search_space[name][idx_val]

        best_val_acc, total_time, _ = train_once(
            params,
            num_epochs=3,
            log_file=f"{model_type}_trust_region_trials.jsonl",
            trial=None,
            enable_pruning=False,
            method_name=f"{model_type}_trust",
            trial_index=t,
            compute_test_metrics=True,
            model_type=model_type,
        )

        results.append({
            "trial": t,
            "value": best_val_acc,
            "time": total_time,
            "params": params,
            "idx_params": idx_params,
        })

    return {"results": results}


# =============== 6. B-NS / Novelty Search-BO ===============
def run_bns(n_trials: int, model_type: str = "cnn"):
    if model_type == "cnn":
        search_space = SEARCH_SPACE_CNN
        param_keys = ["hidden_units", "batch_size", "lr", "weight_decay"]
    elif model_type == "transformer":
        search_space = SEARCH_SPACE_TRANSFORMER
        param_keys = ["embed_dim", "num_heads", "num_layers", "batch_size", "lr", "weight_decay"]
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    def encode_params(p: Dict[str, Any]) -> np.ndarray:
        v = [search_space[key].index(p[key]) for key in param_keys]
        return np.array(v, dtype=float)

    archive_vecs: List[np.ndarray] = []
    results: List[Dict[str, Any]] = []

    for t in range(n_trials):
        num_candidates = 8
        candidates = []
        for _ in range(num_candidates):
            params = {key: search_space[key][np.random.randint(len(search_space[key]))] for key in param_keys}
            vec = encode_params(params)
            if len(archive_vecs) == 0:
                novelty = 1.0
            else:
                dists = [float(np.linalg.norm(vec - v)) for v in archive_vecs]
                novelty = float(np.mean(dists))
            candidates.append((novelty, params, vec))

        candidates.sort(key=lambda x: x[0], reverse=True)
        novelty, chosen_params, chosen_vec = candidates[0]

        best_val_acc, total_time, _ = train_once(
            chosen_params,
            num_epochs=3,
            log_file=f"{model_type}_bns_trials.jsonl",
            trial=None,
            enable_pruning=False,
            method_name=f"{model_type}_bns",
            trial_index=t,
            compute_test_metrics=True,
            model_type=model_type,
        )

        archive_vecs.append(chosen_vec)
        results.append({
            "trial": t,
            "value": best_val_acc,
            "time": total_time,
            "params": chosen_params,
            "novelty": novelty,
        })

    return {"results": results}



# ================== main 入口 ==================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True,
                        choices=["bo", "bohb", "mfbo", "asha", "trust", "bns"])
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--model", type=str, default="transformer", choices=["cnn", "transformer"])
    args = parser.parse_args()

    LOG_DIR.mkdir(exist_ok=True)

    if args.method == "bo":
        study = run_bo(args.trials, args.model)
        print(f"{args.model.upper()} BO best value:", study.best_value, "params:", study.best_params)
    elif args.method == "bohb":
        study = run_bohb(args.trials, model_type=args.model)
        print(f"{args.model.upper()} BOHB best value:", study.best_value, "params:", study.best_params)
    elif args.method == "mfbo":
        study = run_mfbo(args.trials, model_type=args.model)
        print(f"{args.model.upper()} MFBO best value:", study.best_value, "params:", study.best_params)
    elif args.method == "asha":
        study = run_asha(args.trials, model_type=args.model)
        print(f"{args.model.upper()} ASHA best value:", study.best_value, "params:", study.best_params)
    elif args.method == "trust":
        res = run_trust_region(args.trials, model_type=args.model)
        best = max(res["results"], key=lambda r: r["value"])
        print(f"{args.model.upper()} TR best value:", best["value"], "params:", best["params"])
    elif args.method == "bns":
        res = run_bns(args.trials, model_type=args.model)
        best = max(res["results"], key=lambda r: r["value"])
        print(f"{args.model.upper()} BNS best value:", best["value"], "params:", best["params"])


if __name__ == "__main__":
    main()
