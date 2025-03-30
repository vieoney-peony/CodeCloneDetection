import importlib
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

def set_seed(seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def import_object(path):
    # Nhập một đối tượng từ một đường dẫn chuỗi
    module_path, obj_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, obj_name)


# Get the optimizer, scheduler, and scaler
def create_optimizer_scheduler_scaler(config_yaml, model, graph_creator, 
                                      iter_per_epoch=None, total_iter=None):
    # Tạo optimizer, scheduler, và scaler từ cấu hình
    training_config = config_yaml["train"]
    # Optimizer
    optimizer_class = import_object(training_config["optimizer"]["type"])
    optimizer_params = training_config["optimizer"]["params"]
    optimizer = optimizer_class(list(model.parameters()) + list(graph_creator.parameters()), 
                                **optimizer_params)

    # Scheduler
    scheduler = None  # Default value if no scheduler is provided
    if "scheduler" in training_config:  # Check if scheduler is in config
        scheduler_class = import_object(training_config["scheduler"]["type"])
        scheduler_params = training_config["scheduler"]["params"]

        if scheduler_class.__name__ == "get_linear_schedule_with_warmup":
            if iter_per_epoch is None or total_iter is None:
                raise ValueError("total_iter must be provided for linear_schedule")
            scheduler_params["num_warmup_steps"] = iter_per_epoch
            scheduler_params["num_training_steps"] = total_iter
            scheduler = scheduler_class(optimizer, **scheduler_params)
        else:
            scheduler = scheduler_class(optimizer, **scheduler_params)

    # AMP (Automatic Mixed Precision)
    use_amp = training_config.get("amp", False)  # Default to False if not specified
    if use_amp:
        scaler = torch.amp.GradScaler()  # Create GradScaler for AMP
    else:
        scaler = None  # No AMP, no scaler

    return optimizer, scheduler, scaler

def save_checkpoint(model, graph_creator, optimizer, scheduler, scaler, epoch, end_epoch, checkpoint_path):
    state = {
        "epoch": epoch,
        "end_epoch": end_epoch,
        "model_state_dict": model.state_dict(),
        "graph_creator_state_dict": graph_creator.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None
    }
    torch.save(state, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

def load_checkpoint(checkpoint_path, model, graph_creator, optimizer, scheduler, scaler):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    graph_creator.load_state_dict(checkpoint["graph_creator_state_dict"])

    if optimizer and checkpoint["optimizer_state_dict"]:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint["scheduler_state_dict"]:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler and checkpoint["scaler_state_dict"]:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    print(f"Checkpoint loaded from {checkpoint_path}, starting at epoch {checkpoint['epoch'] + 1}")
    return checkpoint["epoch"] + 1, checkpoint["end_epoch"]  # Resume from next epoch

def prepare_batch(batch, idx_map, jsonl_dataset, device):
    # Chuẩn bị dữ liệu cho một batch
    batch_indices_1 = batch['idx1']
    batch_indices_2 = batch['idx2']
    labels = batch['label'] 

    if isinstance(batch_indices_1, torch.Tensor):
        batch_indices_1 = batch_indices_1.tolist()
    if isinstance(batch_indices_2, torch.Tensor):
        batch_indices_2 = batch_indices_2.tolist()

    sorted_indices_1 = [idx_map[idx] for idx in batch_indices_1]
    sorted_indices_2 = [idx_map[idx] for idx in batch_indices_2]

    code_batch_source = jsonl_dataset.select(sorted_indices_1)
    code_batch_target = jsonl_dataset.select(sorted_indices_2)

    if isinstance(labels, list):
        labels = torch.tensor(labels, dtype=torch.float).to(device)
    else:
        labels = labels.to(device, dtype=torch.float)

    return code_batch_source, code_batch_target, labels

def prepare_batchv2(batch, graph_dataset, device):
    # Chuẩn bị dữ liệu cho một batch
    batch_indices_1 = batch['idx1']
    batch_indices_2 = batch['idx2']
    labels = batch['label'] 

    if isinstance(batch_indices_1, torch.Tensor):
        batch_indices_1 = batch_indices_1.tolist()
    if isinstance(batch_indices_2, torch.Tensor):
        batch_indices_2 = batch_indices_2.tolist()

    graph_source = graph_dataset.collate_fn([graph_dataset[idx] for idx in batch_indices_1]).to(device)
    graph_target = graph_dataset.collate_fn([graph_dataset[idx] for idx in batch_indices_2]).to(device)

    if isinstance(labels, list):
        labels = torch.tensor(labels, dtype=torch.float).to(device)
    else:
        labels = labels.to(device, dtype=torch.float)

    return graph_source, graph_target, labels

def save_loss_plot(train_losses, val_losses, file_path):
    plt.figure(figsize=(10, 5))
    plt.plot(
        range(len(train_losses)),
        train_losses,
        label="Train Loss",
        color="blue",
        marker="o",
        linestyle="-",
    )
    plt.plot(
        range(len(val_losses)),
        val_losses,
        label="Validation Loss",
        color="orange",
        marker="o",
        linestyle="-",
    )
    plt.xlabel("Epoch")

    # Chỉnh số lượng tick trên trục x để thưa hơn
    step = max(1, len(train_losses) // 10)  # Cứ mỗi 10% số epoch đặt 1 tick
    plt.xticks(range(0, len(train_losses), step))

    plt.ylabel("Loss")
    plt.title("Training and Validation Loss over Epochs")
    plt.legend()

    plt.savefig(file_path)
    plt.close()