# core_model.py
import time
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# ========== A1. 设备 ==========
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ========== A2. 数据集：FashionMNIST 8:2 划分 ==========
transform = transforms.ToTensor()

full_train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

# 最小规模实验：只用1000个训练样本，200个验证样本
num_train = 1000  # 原来48000
num_val = 200     # 原来12000
train_data, _ = random_split(
    full_train_data,
    [num_train, len(full_train_data) - num_train],
    generator=torch.Generator().manual_seed(42)
)
_, val_data = random_split(
    full_train_data,
    [len(full_train_data) - num_val, num_val],
    generator=torch.Generator().manual_seed(43)
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

class_names = full_train_data.classes
print("Classes:", class_names)

# ========== A3. 模型：TinyVGG 风格 CNN ==========
class FashionMNISTModelV2(nn.Module):
    """
    CNN 模型：
    - 输入 1 通道
    - hidden_units 控制卷积通道数
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units * 7 * 7, output_shape)
        )

    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)
        return x


# ========== A3b. 小型 Transformer 模型 ==========
class FashionMNISTTransformer(nn.Module):
    """
    简化的 Vision Transformer for FashionMNIST
    - 将 28x28 图像展平为 784 序列
    - 使用 patch_size=7 (16x16 patches)
    - 小参数：num_layers=1-2, embed_dim=64-128, num_heads=4-8
    """
    def __init__(self, img_size: int = 28, patch_size: int = 7, embed_dim: int = 64, num_heads: int = 4, num_layers: int = 1, output_shape: int = 10):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 16 patches for 28x28 with patch_size=7

        # Patch embedding
        self.patch_embed = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classifier
        self.classifier = nn.Linear(embed_dim, output_shape)

    def forward(self, x: torch.Tensor):
        # x: [batch, 1, 28, 28]
        x = self.patch_embed(x)  # [batch, embed_dim, 4, 4] -> flatten to [batch, embed_dim, 16]
        x = x.flatten(2).transpose(1, 2)  # [batch, 16, embed_dim]
        x = x + self.pos_embed  # Add position embedding

        # Add class token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  # [batch, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [batch, 17, embed_dim]

        # Transformer
        x = self.transformer(x)  # [batch, 17, embed_dim]

        # Classify using cls token
        cls_output = x[:, 0]  # [batch, embed_dim]
        x = self.classifier(cls_output)  # [batch, 10]
        return x

# ========== A4. accuracy 函数 ==========
def accuracy_fn(y_true: torch.Tensor, y_pred_labels: torch.Tensor) -> float:
    correct = (y_true == y_pred_labels).sum().item()
    return correct / len(y_true) * 100.0


# ========== B1. dataloader 工具 ==========
def get_dataloaders(batch_size: int):
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True
    )
    val_dataloader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False
    )
    return train_dataloader, val_dataloader, test_dataloader


# ========== B2. 单 epoch 训练/验证 ==========
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str
) -> Tuple[float, float]:
    model.train()
    model.to(device)

    running_loss = 0.0
    running_acc = 0.0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        # forward
        y_logits = model(X)
        loss = loss_fn(y_logits, y)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        y_pred_labels = y_logits.argmax(dim=1)
        running_acc += accuracy_fn(y_true=y, y_pred_labels=y_pred_labels)

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = running_acc / len(dataloader)
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: str
) -> Tuple[float, float]:
    model.eval()
    model.to(device)

    running_loss = 0.0
    running_acc = 0.0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        y_logits = model(X)
        loss = loss_fn(y_logits, y)

        running_loss += loss.item()
        y_pred_labels = y_logits.argmax(dim=1)
        running_acc += accuracy_fn(y_true=y, y_pred_labels=y_pred_labels)

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = running_acc / len(dataloader)
    return epoch_loss, epoch_acc


# ========== B3. 通用 train_once（记录尽可能多信息） ==========
LOG_DIR = Path("bo_logs")
LOG_DIR.mkdir(exist_ok=True)

def train_once(
    params: Dict[str, Any],
    num_epochs: int = 3,
    log_file: Optional[str] = None,
    trial: "Optional[object]" = None,     # 给 Optuna 用
    enable_pruning: bool = False,
    method_name: Optional[str] = None,
    trial_index: Optional[int] = None,
    compute_test_metrics: bool = True,
    model_type: str = "cnn",  # 新增：选择模型类型 "cnn" 或 "transformer"
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """
    统一的训练封装：
    - params: 必含 {hidden_units, lr, batch_size, weight_decay}
    - 返回: best_val_acc, total_time_sec, history(逐 epoch 指标)
    - 日志中记录：
        * 方法名 / trial index
        * best / final train/val
        * test acc/loss
        * 时间信息
        * pruning 信息
        * 逐 epoch history + 每个 epoch 用时
    """
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=params["batch_size"])
    
    if model_type == "cnn":
        model = FashionMNISTModelV2(
            input_shape=1,
            hidden_units=params["hidden_units"],
            output_shape=len(class_names)
        ).to(device)
    elif model_type == "transformer":
        model = FashionMNISTTransformer(
            embed_dim=params["embed_dim"],
            num_heads=params["num_heads"],
            num_layers=params["num_layers"],
            output_shape=len(class_names)
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"]
    )

    history: List[Dict[str, Any]] = []
    best_val_acc = 0.0
    best_val_loss = float("inf")
    start_time = time.time()

    pruned = False
    pruned_epoch: Optional[int] = None

    for epoch in range(num_epochs):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device
        )
        val_loss, val_acc = evaluate(
            model, val_loader, loss_fn, device
        )

        epoch_time = time.time() - epoch_start

        if val_acc > best_val_acc:
            best_val_acc = val_acc
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        record_epoch = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "epoch_time_sec": epoch_time,
        }
        history.append(record_epoch)

        # pruning 逻辑
        if trial is not None and enable_pruning:
            import optuna
            trial.report(val_acc, step=epoch)
            if trial.should_prune():
                pruned = True
                pruned_epoch = epoch
                break

    total_time = time.time() - start_time
    num_epochs_run = len(history)

    # test 指标（用最后一版模型）
    if compute_test_metrics:
        test_loss, test_acc = evaluate(
            model, test_loader, loss_fn, device
        )
    else:
        test_loss, test_acc = None, None

    if history:
        final_train_loss = history[-1]["train_loss"]
        final_train_acc = history[-1]["train_acc"]
        final_val_loss = history[-1]["val_loss"]
        final_val_acc = history[-1]["val_acc"]
    else:
        # 理论上不会出现，但防守一下
        final_train_loss = final_train_acc = final_val_loss = final_val_acc = None

    avg_epoch_time = total_time / max(1, num_epochs_run)

    # 记录 JSONL
    if log_file is not None:
        path = LOG_DIR / log_file
        with path.open("a", encoding="utf-8") as f:
            record = {
                "method": method_name,
                "trial_index": trial_index,
                "params": params,
                "best_val_acc": best_val_acc,
                "best_val_loss": best_val_loss,
                "final_train_acc": final_train_acc,
                "final_train_loss": final_train_loss,
                "final_val_acc": final_val_acc,
                "final_val_loss": final_val_loss,
                "test_acc": test_acc,
                "test_loss": test_loss,
                "total_time_sec": total_time,
                "avg_epoch_time_sec": avg_epoch_time,
                "num_epochs_run": num_epochs_run,
                "planned_num_epochs": num_epochs,
                "pruned": pruned,
                "pruned_epoch": pruned_epoch,
                "history": history,
            }
            f.write(json.dumps(record) + "\n")

    # 给 Optuna 的 user_attr（可选）
    if trial is not None:
        trial.set_user_attr("total_time_sec", total_time)
        trial.set_user_attr("num_epochs_run", num_epochs_run)
        trial.set_user_attr("pruned", pruned)

    # 如果被 prune，通知 Optuna
    if pruned and trial is not None and enable_pruning:
        import optuna
        raise optuna.TrialPruned()

    return best_val_acc, total_time, history
