import os
import argparse
from tqdm import tqdm

import numpy as np
import torch

from config import Config
from modules import build_model, inference
from dataset import build_dataset
from loss import bce
from utils import set_seed, create_optimizer_scheduler_scaler, \
                    save_checkpoint, load_checkpoint, prepare_batch

from sklearn.metrics import precision_score, recall_score, f1_score
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval(model, graph_creator, jsonl_dataset, 
            idx_map, val_loader, batch_size, device):
    model.eval()
    graph_creator.eval()
    total_loss = 0
    logits=[]  
    y_trues=[]
    
    pbar = tqdm(enumerate(val_loader), 
                total=len(val_loader), 
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
                desc="Validating")

    for i, batch in pbar:
        with torch.no_grad():
            code_batch_source, code_batch_target, labels = prepare_batch(batch, idx_map, jsonl_dataset, device)
            logit = inference(graph_creator, model, code_batch_source, code_batch_target)

            loss = bce(logit, labels)

            total_loss = (total_loss*i + loss.item()) / (i+1)

            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
    
    logits=np.concatenate(logits,0)
    y_trues=np.concatenate(y_trues,0)
    best_threshold=0
    best_f1=0
    
    for i in range(1,100):
        threshold=i/100
        y_preds=logits[:]>threshold
        recall=recall_score(y_trues, y_preds, zero_division=0.0)
        precision=precision_score(y_trues, y_preds, zero_division=0.0)
        f1=f1_score(y_trues, y_preds, zero_division=0.0) 
        if f1>best_f1:
            best_f1=f1
            best_threshold=threshold

    y_preds=logits[:]>best_threshold

    recall=recall_score(y_trues, y_preds, zero_division=0.0)
    precision=precision_score(y_trues, y_preds, zero_division=0.0)
    f1=f1_score(y_trues, y_preds, zero_division=0.0)             

    result = {
        "eval_loss": total_loss,
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_threshold": best_threshold,
    }

    print("***** Eval results *****")
    for key in sorted(result.keys()):
        print("  {} = {}".format(key, round(result[key], 4)))

    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--split", type=str, choices=["valid", "test"], default="test", help="Split to evaluate")
    args = parser.parse_args()