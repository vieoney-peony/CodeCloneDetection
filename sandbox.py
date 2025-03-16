import torch
import torch.nn as nn
from torch_geometric.data import Data

# Giả sử original_embed là tensor learnable
original_embed = torch.randn(10, 128, requires_grad=True)  # (10 nodes, 128 dim)

# Biến đổi embedding thành node_features bằng Linear (Differentiable)
linear_proj = nn.Linear(128, 256)  
node_features = linear_proj(original_embed)  # (10, 256) -> vẫn track gradient

# Định nghĩa đồ thị với edge_index
edge_index_1 = torch.tensor([[0, 1, 2], [1, 2, 0]])  # Đồ thị giả lập

# Đóng gói thành batch graph
batch_graph = Data(x=node_features, edge_index=edge_index_1)

# Forward qua GNN
gnn = nn.Linear(256, 1)  # Giả sử GNN đơn giản là Linear
output = gnn(batch_graph.x)  

# Backward
loss = output.mean()
loss.backward()

print(original_embed.grad)  # ✅ Không phải None, có gradient
