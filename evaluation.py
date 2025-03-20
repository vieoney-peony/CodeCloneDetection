import os
import argparse
from tqdm import tqdm

import numpy as np
import torch

from config import Config
from modules import build_model, inference
from dataset import build_dataset
from loss import bce, cosine_similarity_loss
from metrics import calculate_metrics
from utils import set_seed, create_optimizer_scheduler_scaler, \
                    save_checkpoint, load_checkpoint, prepare_batch


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
            
            loss = cosine_similarity_loss(logit, labels)

            total_loss = (total_loss*i + loss.item()) / (i+1)

            logits.append(logit.sigmoid().cpu().numpy())
            y_trues.append(labels.cpu().numpy())
    
    logits=np.concatenate(logits,0)
    y_trues=np.concatenate(y_trues,0)       
    result = calculate_metrics(logits, y_trues)
    result["eval_loss"] = total_loss

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