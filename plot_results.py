# plot_results.py
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt

LOG_DIR = Path("bo_logs")

METHOD_TO_FILE = {
    "bo": "transformer_bo_trials.jsonl",
    "bohb": "transformer_bohb_trials.jsonl",
    "mfbo": "transformer_mfbo_trials.jsonl",
    "asha": "transformer_asha_trials.jsonl",
    "trust": "transformer_trust_region_trials.jsonl",
    "bns": "transformer_bns_trials.jsonl",
}


def load_jsonl(path: Path) -> List[Dict]:
    records = []
    if not path.exists():
        print("File not found:", path)
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def extract_curves(records: List[Dict]):
    """
    从 jsonl 记录中提取：
    - trial-wise best_acc
    - trial-wise time
    然后计算：
    - best_so_far_acc vs trial
    - best_so_far_acc vs cumulative_time
    """
    trial_acc = []
    trial_time = []

    for rec in records:
        trial_acc.append(rec.get("best_val_acc", 0.0))
        trial_time.append(rec.get("total_time_sec", 0.0))

    trial_acc = np.array(trial_acc, dtype=float)
    trial_time = np.array(trial_time, dtype=float)

    best_so_far = np.maximum.accumulate(trial_acc)
    cumulative_time = np.cumsum(trial_time)

    return trial_acc, best_so_far, trial_time, cumulative_time


def collect_per_method_stats(records: List[Dict]) -> Dict[str, np.ndarray]:
    """提取各种可画图的统计量"""
    best_acc = []
    total_time = []
    lr = []
    batch_size = []
    hidden_units = []
    weight_decay = []
    final_train_acc = []
    final_val_acc = []
    test_acc = []
    overfit_gap = []

    for rec in records:
        best_acc.append(rec.get("best_val_acc", 0.0))
        total_time.append(rec.get("total_time_sec", 0.0))

        params = rec.get("params", {})
        lr.append(params.get("lr", 0.0))
        batch_size.append(params.get("batch_size", 0))
        hidden_units.append(params.get("hidden_units", 0))
        weight_decay.append(params.get("weight_decay", 0.0))

        ft = rec.get("final_train_acc", None)
        fv = rec.get("final_val_acc", None)
        fa = rec.get("test_acc", None)

        final_train_acc.append(ft if ft is not None else np.nan)
        final_val_acc.append(fv if fv is not None else np.nan)
        test_acc.append(fa if fa is not None else np.nan)

        if ft is not None and fv is not None:
            overfit_gap.append(ft - fv)
        else:
            overfit_gap.append(np.nan)

    return {
        "best_acc": np.array(best_acc, float),
        "total_time": np.array(total_time, float),
        "lr": np.array(lr, float),
        "batch_size": np.array(batch_size, float),
        "hidden_units": np.array(hidden_units, float),
        "weight_decay": np.array(weight_decay, float),
        "final_train_acc": np.array(final_train_acc, float),
        "final_val_acc": np.array(final_val_acc, float),
        "test_acc": np.array(test_acc, float),
        "overfit_gap": np.array(overfit_gap, float),
    }


def plot_all():
    methods = ["bo", "bohb", "mfbo", "asha", "trust", "bns"]

    curves = {}
    stats = {}

    for m in methods:
        file = METHOD_TO_FILE[m]
        recs = load_jsonl(LOG_DIR / file)
        if len(recs) == 0:
            print(f"[WARN] method {m} has no records, skip.")
            continue
        trial_acc, best_so_far, trial_time, cumulative_time = extract_curves(recs)
        curves[m] = {
            "trial_acc": trial_acc,
            "best_so_far": best_so_far,
            "trial_time": trial_time,
            "cumulative_time": cumulative_time,
        }
        stats[m] = collect_per_method_stats(recs)

    # 1) Best Accuracy vs Trial
    plt.figure()
    for m, c in curves.items():
        plt.plot(np.arange(1, len(c["best_so_far"]) + 1), c["best_so_far"], label=m)
    plt.xlabel("Trial")
    plt.ylabel("Best Val Accuracy (%)")
    plt.title("Best Accuracy vs Trial")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("new_pics/best_acc_vs_trial.png", dpi=200)

    # 2) Best Accuracy vs Cumulative Time
    plt.figure()
    for m, c in curves.items():
        plt.plot(c["cumulative_time"], c["best_so_far"], label=m)
    plt.xlabel("Cumulative Time (s)")
    plt.ylabel("Best Val Accuracy (%)")
    plt.title("Best Accuracy vs Cumulative Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("new_pics/best_acc_vs_time.png", dpi=200)

    # 3) Individual Trial Accuracy
    plt.figure()
    for m, c in curves.items():
        plt.plot(np.arange(1, len(c["trial_acc"]) + 1), c["trial_acc"], label=m, marker="o", linestyle="--", alpha=0.7)
    plt.xlabel("Trial")
    plt.ylabel("Trial Best Val Accuracy (%)")
    plt.title("Individual Trial Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("new_pics/individual_trial_acc.png", dpi=200)

    # 4) Best Acc histogram
    plt.figure()
    for m, s in stats.items():
        if m not in stats:
            continue
        plt.hist(s["best_acc"], bins=10, alpha=0.5, label=m)
    plt.xlabel("Best Val Accuracy (%)")
    plt.ylabel("Count")
    plt.title("Histogram of Best Val Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("new_pics/best_acc_hist.png", dpi=200)

    # 5) Time histogram
    plt.figure()
    for m, s in stats.items():
        plt.hist(s["total_time"], bins=10, alpha=0.5, label=m)
    plt.xlabel("Total Time per Trial (s)")
    plt.ylabel("Count")
    plt.title("Histogram of Trial Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig("new_pics/time_hist.png", dpi=200)

    # 6) lr vs acc
    plt.figure()
    for m, s in stats.items():
        x = np.log10(s["lr"] + 1e-12)
        plt.scatter(x, s["best_acc"], label=m, alpha=0.7)
    plt.xlabel("log10(lr)")
    plt.ylabel("Best Val Accuracy (%)")
    plt.title("Learning Rate vs Best Val Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("new_pics/lr_vs_acc.png", dpi=200)

    # 7) batch_size vs acc
    plt.figure()
    for m, s in stats.items():
        plt.scatter(s["batch_size"], s["best_acc"], label=m, alpha=0.7)
    plt.xlabel("Batch Size")
    plt.ylabel("Best Val Accuracy (%)")
    plt.title("Batch Size vs Best Val Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("new_pics/batch_vs_acc.png", dpi=200)

    # 8) hidden_units vs acc
    plt.figure()
    for m, s in stats.items():
        plt.scatter(s["hidden_units"], s["best_acc"], label=m, alpha=0.7)
    plt.xlabel("Hidden Units")
    plt.ylabel("Best Val Accuracy (%)")
    plt.title("Hidden Units vs Best Val Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("new_pics/hidden_vs_acc.png", dpi=200)

    # 9) weight_decay vs acc
    plt.figure()
    for m, s in stats.items():
        x = np.log10(s["weight_decay"] + 1e-12)
        plt.scatter(x, s["best_acc"], label=m, alpha=0.7)
    plt.xlabel("log10(weight_decay)")
    plt.ylabel("Best Val Accuracy (%)")
    plt.title("Weight Decay vs Best Val Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("new_pics/wd_vs_acc.png", dpi=200)

    # 10) overfitting gap vs best acc
    plt.figure()
    for m, s in stats.items():
        plt.scatter(s["overfit_gap"], s["best_acc"], label=m, alpha=0.7)
    plt.xlabel("Final (Train Acc - Val Acc) (%)")
    plt.ylabel("Best Val Accuracy (%)")
    plt.title("Overfitting Gap vs Best Val Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("new_pics/overfit_vs_acc.png", dpi=200)

    # 11) val vs test acc
    plt.figure()
    for m, s in stats.items():
        plt.scatter(s["final_val_acc"], s["test_acc"], label=m, alpha=0.7)
    plt.xlabel("Final Val Accuracy (%)")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Final Val vs Test Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("new_pics/val_vs_test_acc.png", dpi=200)

    print("Saved figures:")
    print("- new_pics/best_acc_vs_trial.png")
    print("- new_pics/best_acc_vs_time.png")
    print("- new_pics/individual_trial_acc.png")
    print("- new_pics/best_acc_hist.png")
    print("- new_pics/time_hist.png")
    print("- new_pics/lr_vs_acc.png")
    print("- new_pics/batch_vs_acc.png")
    print("- new_pics/hidden_vs_acc.png")
    print("- new_pics/wd_vs_acc.png")
    print("- new_pics/overfit_vs_acc.png")
    print("- new_pics/val_vs_test_acc.png")


if __name__ == "__main__":
    plot_all()
