import os
import argparse
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast

from config import Config
from modules import build_modelv2, inferencev2
from dataset import build_dataset
from loss import bce, cosine_similarity_loss, ce
from metrics import calculate_metrics
from utils import set_seed, load_checkpoint, prepare_batch, prepare_batchv2

set_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval(model, graph_creator, jsonl_dataset, 
        graph_dataset, idx_map, val_loader, batch_size, 
        device, max_iter=None):
    model.eval()
    graph_creator.eval()
    total_loss = 0
    logits=[]  
    y_trues=[]
    
    if max_iter is not None:
        pbar = tqdm(enumerate(val_loader), 
                    total=min(max_iter, len(val_loader)),
                    bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
                    desc="Validating")
    else:
        pbar = tqdm(enumerate(val_loader), 
                    total=len(val_loader), 
                    bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
                    desc="Validating")

    for i, batch in pbar:
        if max_iter is not None and i >= max_iter:
            break
        with torch.no_grad():
            with autocast(device_type=device.type):
                code_batch_source, code_batch_target, labels = prepare_batchv2(batch, graph_dataset, device)
                logit = inferencev2(graph_creator, model, code_batch_source, code_batch_target)
                
                # loss = cosine_similarity_loss(logit, labels)
                loss = ce(logit, labels.long())

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

def main(args):
    config = Config(args.config)

    graph_creator, model = build_modelv2(config)
    model = model.to(device)
    graph_creator = graph_creator.to(device)

    if args.checkpoint is not None:
        load_checkpoint(args.checkpoint, model, graph_creator, None, None, None)

    jsonl_dataset, txt_dataset, graph_dataset = build_dataset(config)

    idx_map = {v: i for i, v in enumerate(jsonl_dataset['idx'])}
    
    test_txt = txt_dataset[args.split]
    test_loader = DataLoader(test_txt, batch_size=config["dataset"]["batch_size"], shuffle=False)

    eval(model, graph_creator, jsonl_dataset, graph_dataset,
        idx_map, test_loader, None, device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--split", type=str, choices=["valid", "test"], default="test", help="Split to evaluate")
    args = parser.parse_args()

    main(args)