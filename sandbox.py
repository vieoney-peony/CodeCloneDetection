import torch

node_embs = torch.zeros((5, 3))  # 5 nodes, mỗi node có embedding dim=3

inverse_indices = torch.tensor([2, 0, 1, 3, 2])  # Node 2 xuất hiện 2 lần
some_values = torch.tensor([
    [0.1, 0.2, 0.3],  # Giá trị cho node 2
    [0.4, 0.5, 0.6],  # Giá trị cho node 0
    [0.7, 0.8, 0.9],  # Giá trị cho node 1
    [1.0, 1.1, 1.2],  # Giá trị cho node 3
    [1.3, 1.4, 1.5],  # Giá trị cho node 2 (Ghi đè lên giá trị trước đó)
])

node_embs[inverse_indices] = some_values
print(node_embs)
