import torch
import torch.nn as nn
import torch.nn.functional as F

bce = nn.BCEWithLogitsLoss()

mse = nn.MSELoss()

rmse = lambda x, y: torch.sqrt(mse(x, y))