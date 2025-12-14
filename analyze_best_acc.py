import os
import json
from typing import Dict, Any, Optional, List

# 日志目录
LOG_DIR = "bo_logs"

# 方法列表
METHODS = ["transformer_bo", "transformer_bohb", "transformer_mfbo", "transformer_asha", "transformer_trust", "transformer_bns"]

# 不同方法对应的日志文件名
LOG_FILES = {
    "transformer_bo": "transformer_bo_trials.jsonl",
    "transformer_bohb": "transformer_bohb_trials.jsonl",
    "transformer_mfbo": "transformer_mfbo_trials.jsonl",
    "transformer_asha": "transformer_asha_trials.jsonl",
    "transformer_trust": "transformer_trust_region_trials.jsonl",
    "transformer_bns": "transformer_bns_trials.jsonl",
}


def load_best_for_method(method: str) -> Optional[Dict[str, Any]]:
    """
    读取某个 method 的 jsonl 日志，返回该方法的最佳 trial 信息：
    - best_val_acc
    - trial index/id
    - params
    """
    filename = LOG_FILES.get(method)
    if filename is None:
        print(f"[{method}] no log filename configured.")
        return None

    log_path = os.path.join(LOG_DIR, filename)
    if not os.path.exists(log_path):
        print(f"[{method}] log file not found: {log_path}")
        return None

    best_acc: Optional[float] = None
    best_trial: Optional[Any] = None
    best_params: Optional[Dict[str, Any]] = None

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            # 找准确率字段
            acc = (
                rec.get("best_val_acc")
                or rec.get("val_acc")
                or rec.get("val_accuracy")
            )
            if acc is None:
                continue

            # 更新最佳
            if best_acc is None or acc > best_acc:
                best_acc = acc
                # trial 可能用不同字段名，统一兜底
                best_trial = (
                    rec.get("trial_index")
                    or rec.get("trial_number")
                    or rec.get("trial")
                    or rec.get("trial_id")
                )
                best_params = rec.get("params")

    if best_acc is None:
        print(f"[{method}] no accuracy field found in {log_path}")
        return None

    return {
        "method": method,
        "best_val_acc": best_acc,
        "best_trial": best_trial,
        "best_params": best_params,
    }


def main():
    print("==== Best validation accuracy per method ====")
    results: List[Dict[str, Any]] = []

    for m in METHODS:
        info = load_best_for_method(m)
        if info is None:
            continue
        results.append(info)
        acc = info["best_val_acc"]
        trial = info["best_trial"]
        params = info["best_params"]
        print(
            f"{m:5s} | best_val_acc = {acc:.3f} "
            f"| trial = {trial} "
            f"| params = {params}"
        )

    if not results:
        print("\nNo valid results found.")
        return

    # 全局最优
    best_overall = max(results, key=lambda x: x["best_val_acc"])
    print("\n==== Overall best across all methods ====")
    print(
        f"Method = {best_overall['method']}, "
        f"best_val_acc = {best_overall['best_val_acc']:.3f}, "
        f"trial = {best_overall['best_trial']}, "
        f"params = {best_overall['best_params']}"
    )


if __name__ == "__main__":
    main()
