import torch
import torch.nn as nn
import torch.nn.functional as F

bce = nn.BCEWithLogitsLoss()

mse = nn.MSELoss()

rmse = lambda x, y: torch.sqrt(mse(x, y))

def cosine_similarity_loss(sim_score, label, margin=0.0):
    y = 2 * label - 1  # Chuyển nhãn từ {0,1} thành {-1,1}
    loss_pos = 1 - sim_score  # Loss khi y = 1
    loss_neg = torch.clamp(sim_score - margin, min=0)  # Loss khi y = -1

    loss = torch.where(y == 1, loss_pos, loss_neg)  # Chọn loss theo y
    # print(y.tolist())
    # print(sim_score.tolist())
    # print(loss.tolist())
    return loss.mean()