import os
import argparse
import pytz
import datetime
from tqdm import tqdm
import random

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast

from config import Config
from dataset import build_dataset, PosNegSampler
from modules import build_model, inference, build_modelv2, inferencev2
from loss import bce, cosine_similarity_loss, ce
from evaluation import eval

from utils import set_seed, create_optimizer_scheduler_scaler, \
                    save_checkpoint, load_checkpoint, prepare_batchv2,\
                    prepare_batch, save_loss_plot

set_seed(0)

from torch.utils.tensorboard import SummaryWriter

vietnam_tz = pytz.timezone("Asia/Ho_Chi_Minh")
__current_time__ = datetime.datetime.now(vietnam_tz).strftime("%Y-%m-%d_%H-%M-%S")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init(config):
    graph_creator, model = build_modelv2(config)
    jsonl_dataset, txt_dataset, graph_dataset = build_dataset(config)

    idx_map = {v: i for i, v in enumerate(jsonl_dataset['idx'])}

    train_txt = txt_dataset['train']
    val_txt = txt_dataset['valid']

    # Chọn ngẫu nhiên 5% dữ liệu từ tập validation để kiểm tra
    random_indices = random.sample(range(len(val_txt)), int(len(val_txt)*0.1))
    val_txt = val_txt.select(random_indices)

    train_loader = DataLoader(train_txt, batch_size=config["dataset"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_txt, batch_size=config["dataset"]["batch_size"], shuffle=False)

    pos_neg_sampler = PosNegSampler(train_txt)
    
    iter_per_epoch = len(train_loader)
    if config["train"]["max_iter"] is not None:
        iter_per_epoch = min(len(train_loader), config["train"]["max_iter"])
    total_iter = config["train"]["epochs"] * iter_per_epoch

    optimizer, scheduler, scaler = create_optimizer_scheduler_scaler(config, model, graph_creator,
                                                                     iter_per_epoch=iter_per_epoch,
                                                                     total_iter=total_iter)

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

    return model, graph_creator, jsonl_dataset, graph_dataset, \
            idx_map, train_loader, val_loader, pos_neg_sampler, \
            optimizer, scheduler, scaler, \
            start_epoch, end_epoch, writer


def train_one_epoch(model, graph_creator, jsonl_dataset, 
                    graph_dataset, idx_map, train_loader, pos_neg_sampler, batch_size,
                    optimizer, scheduler, scaler, max_iter=None):
    model.train()
    graph_creator.train()
    total_loss = 0

    if max_iter is not None:
        pbar = tqdm(
            enumerate(train_loader),
            desc="Training",
            total=min(max_iter, len(train_loader)),
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        )
    else:
        pbar = tqdm(
            enumerate(train_loader),
            desc="Training",
            total=len(train_loader),
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        )

    for i, batch in pbar:
        if max_iter is not None and i >= max_iter:
            break
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        
        graph_source, graph_target, labels = prepare_batchv2(batch, graph_dataset, device)
        
        # get positive and negative samples
        # pos_code_batch_source, pos_code_batch_target, pos_labels = prepare_batch(pos_batch, idx_map, jsonl_dataset, device)
        # neg_code_batch_source, neg_code_batch_target, neg_labels = prepare_batch(neg_batch, idx_map, jsonl_dataset, device)
        # pos_neg_labels = torch.cat([pos_labels, neg_labels], dim=0)

        with autocast(device_type=device.type, enabled=scaler is not None):
            logit = inferencev2(graph_creator, model, graph_source, graph_target)

            # pos_logit = inference(graph_creator, model, pos_code_batch_source, pos_code_batch_target)
            # neg_logit = inference(graph_creator, model, neg_code_batch_source, neg_code_batch_target)

            # pos_neg_logit = torch.cat([pos_logit, neg_logit], dim=0)
            
            # loss calculation
            loss = ce(logit, labels.long())
            
            loss2 = 0
            # loss2 = cosine_similarity_loss(pos_neg_logit, pos_neg_labels)
            # loss2 = ce(pos_logit, pos_labels.long())
            # loss2 = cosine_similarity_loss(neg_logit, neg_labels)

        total_loss = (total_loss*i + loss.item()) / (i+1)

        # Backward pass
        scaler.scale(loss + loss2).backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + 
                                       list(graph_creator.parameters()), 
                                       max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
            scheduler.step()
            # print(f"Scheduler step: {scheduler.get_last_lr()}")

        pbar.set_postfix({"Train loss": f"{total_loss:.6f}"})

    if scheduler is not None and \
        not isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
        scheduler.step()

    return total_loss
    
    

def train(arg):
    config = Config(arg.config)

    model, graph_creator, jsonl_dataset, graph_dataset, \
    idx_map, train_loader, val_loader, pos_neg_sampler, \
    optimizer, scheduler, scaler, \
    start_epoch, end_epoch, writer = init(config)

    log_dir = writer.log_dir
    batch_size = config["dataset"]["batch_size"]
    max_iter = config["train"]["max_iter"]

    best_val_loss = float("inf")
    last_model_path = os.path.join(log_dir, "last.pt")
    best_model_path = os.path.join(log_dir, "best.pt")
    loss_plot_path = os.path.join(log_dir, "loss_plot.png")

    train_losses = []
    val_losses = []

    for epoch in range(start_epoch, end_epoch):
        print(f"Epoch {epoch+1}/{end_epoch}")
        t_loss = train_one_epoch(model, graph_creator, jsonl_dataset, graph_dataset,
                                 idx_map, train_loader, pos_neg_sampler, batch_size,
                                 optimizer, scheduler, scaler, max_iter)

        result = eval(model, graph_creator, jsonl_dataset, graph_dataset,
                       idx_map, val_loader, batch_size, device)
        
        # saving
        if result['eval_loss'] < best_val_loss:
            best_val_loss = result['eval_loss']
            save_checkpoint(model, graph_creator, optimizer, scheduler, scaler, epoch, end_epoch, best_model_path)

        save_checkpoint(model, graph_creator, optimizer, scheduler, scaler, epoch, end_epoch, last_model_path)

        train_losses.append(t_loss)
        val_losses.append(result["eval_loss"])

        writer.add_scalar("Train/loss", t_loss, epoch)
        writer.add_scalar("Eval/loss", result["eval_loss"], epoch)
        writer.add_scalar("Eval/F1", result["eval_f1"], epoch)
        writer.add_scalar("Eval/Precision", result["eval_precision"], epoch)
        writer.add_scalar("Eval/Recall", result["eval_recall"], epoch)

        save_loss_plot(train_losses, val_losses, loss_plot_path)
        
        print("-" * 50)
        
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument("--resume", type=bool, default=False, help="Resume training")
    args = parser.parse_args()

    train(args)