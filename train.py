import os
import argparse
import pytz
import datetime
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast

from config import Config
from dataset import build_dataset, PosNegSampler
from modules import build_model, inference

from utils import set_seed, create_optimizer_scheduler_scaler, \
                    save_checkpoint, load_checkpoint, prepare_batch

set_seed(0)

from torch.utils.tensorboard import SummaryWriter

vietnam_tz = pytz.timezone("Asia/Ho_Chi_Minh")
__current_time__ = datetime.datetime.now(vietnam_tz).strftime("%Y-%m-%d_%H-%M-%S")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init(config):
    graph_creator, model = build_model(config)
    jsonl_dataset, txt_dataset = build_dataset(config)

    idx_map = {v: i for i, v in enumerate(jsonl_dataset['idx'])}

    train_txt = txt_dataset['train']
    val_txt = txt_dataset['valid']

    train_loader = DataLoader(train_txt, batch_size=config["dataset"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_txt, batch_size=config["dataset"]["batch_size"], shuffle=False)

    pos_neg_sampler = PosNegSampler(train_txt)

    optimizer, scheduler, scaler = create_optimizer_scheduler_scaler(config, model, graph_creator)

    # Mặc định train từ epoch 0
    start_epoch = 0
    end_epoch = config["train"]["epochs"]

    if args.checkpoint is not None:
        print("Loading checkpoint...")

        if args.resume:
            # Load toàn bộ checkpoint để tiếp tục training từ epoch trước
            start_epoch, end_epoch = load_checkpoint(
                args.checkpoint, model, graph_creator, optimizer, scheduler, scaler
            )
            print(f"Resuming training from epoch {start_epoch}")

        else:
            # Chỉ load model, không ảnh hưởng optimizer/scheduler, bắt đầu lại từ epoch 0
            load_checkpoint(args.checkpoint, model, graph_creator, None, None, None)
            print("Loaded pre-trained model, starting training from epoch 0")
    else:
        if args.resume:
            raise ValueError("Cannot resume training without a checkpoint")

    model.to(device)
    graph_creator.to(device)


    os.makedirs(config["log_dir"], exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(config["log_dir"], f"train_{__current_time__}"))

    return model, graph_creator, jsonl_dataset, \
            idx_map, train_loader, val_loader, pos_neg_sampler, \
            optimizer, scheduler, scaler, \
            start_epoch, end_epoch, writer


def train_one_epoch(model, graph_creator, jsonl_dataset, 
                    idx_map, train_loader, pos_neg_sampler, batch_size,
                    optimizer, scheduler, scaler):
    total_loss = 0

    for batch in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()
        pos_batch, neg_batch = pos_neg_sampler.sample(batch_size)
        # get default samples
        code_batch_source, code_batch_target, labels = prepare_batch(batch, idx_map, jsonl_dataset, device)
        
        # get positive and negative samples
        pos_code_batch_source, pos_code_batch_target, pos_labels = prepare_batch(pos_batch, idx_map, jsonl_dataset, device)
        neg_code_batch_source, neg_code_batch_target, neg_labels = prepare_batch(neg_batch, idx_map, jsonl_dataset, device)
        pos_neg_labels = torch.cat([pos_labels, neg_labels], dim=0)
        
        with autocast(device_type=device.type, enabled=scaler is not None):
            scores = inference(graph_creator, model, code_batch_source, code_batch_target)

            pos_scores = inference(graph_creator, model, pos_code_batch_source, pos_code_batch_target)
            neg_scores = inference(graph_creator, model, neg_code_batch_source, neg_code_batch_target)

            pos_neg_scores = torch.cat([pos_scores, neg_scores], dim=0)

            loss = torch.nn.functional.binary_cross_entropy_with_logits(scores, labels)
            loss2 = torch.nn.functional.binary_cross_entropy_with_logits(pos_neg_scores, pos_neg_labels)

        total_loss += loss.item()

        # Backward pass
        scaler.scale(loss + loss2).backward()
        scaler.step(optimizer)
        scaler.update()

    if scheduler is not None:
        scheduler.step()

    return total_loss / len(train_loader)
    
    

def train(arg):
    config = Config(arg.config)

    model, graph_creator, jsonl_dataset, \
    idx_map, train_loader, val_loader, pos_neg_sampler, \
    optimizer, scheduler, scaler, \
    start_epoch, end_epoch, writer = init(config)

    batch_size = config["dataset"]["batch_size"]

    for epoch in range(start_epoch, end_epoch):
        print(f"Epoch {epoch+1}/{end_epoch}")
        model.train()
        graph_creator.train()
        t_loss = train_one_epoch(model, graph_creator, jsonl_dataset, 
                                 idx_map, train_loader, pos_neg_sampler, batch_size,
                                 optimizer, scheduler, scaler)

        break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument("--resume", type=bool, default=False, help="Resume training")
    args = parser.parse_args()

    train(args)