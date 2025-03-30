import torch


node_embs = torch.zeros((5, 3)).to('cuda')  # 5 nodes, mỗi node có embedding dim=3

node_embs_copy = node_embs.clone()  # Tạo một bản sao của node_embs
print(node_embs_copy.device)  # Kiểm tra device của node_embs_copy
