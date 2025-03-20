import numpy as np

from sklearn.metrics import precision_score, recall_score, f1_score

def calculate_metrics(logits:np.ndarray, y_trues: np.ndarray):
    best_threshold=0
    best_f1=0
    
    for i in range(-100, 100):
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
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_threshold": best_threshold,
    }
    return result