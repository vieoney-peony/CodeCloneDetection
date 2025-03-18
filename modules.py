import os
from typing import Tuple, Union

import torch
from torch import Tensor
import torch.nn as nn

import math
import json

from datasets import load_from_disk
from order_flow_ast import JavaASTGraphVisitor, JavaASTLiteralNode, JavaASTBinaryOpNode
from transformers import AutoTokenizer, AutoModel

from torch_geometric.data import HeteroData, Batch
from torch_geometric.nn import HeteroConv, MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_geometric.utils import spmm

from utils import set_seed

set_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sin_cos_encoding(pos: torch.tensor, d_model):
    if pos.dim() == 1:  # Nếu pos là tensor 1D [num_positions]
        pos = pos.unsqueeze(1)  # Chuyển thành [num_positions, 1]

    num_positions = pos.shape[0]  # Số lượng vị trí
    pe = torch.zeros(num_positions, d_model, device=pos.device)  # Tensor output
    
    div_term = torch.exp(torch.arange(0, d_model, 2, device=pos.device) * (-math.log(10000.0) / d_model))

    # Áp dụng sin/cos, mở rộng `pos` để broadcast với `div_term`
    pe[:, 0::2] = torch.sin(pos * div_term)  # Sin encoding
    pe[:, 1::2] = torch.cos(pos * div_term)  # Cos encoding
    return pe

def check_node_embedding(order, node_edge_embedding):
    node_emb_dict = {}

    for i, (src, _, tgt) in enumerate(order):
        node_emb_src = node_edge_embedding[i, 0, :]  # Source node embedding
        node_emb_tgt = node_edge_embedding[i, 2, :]  # Target node embedding

        # Kiểm tra nếu node đã xuất hiện trước đó
        if src in node_emb_dict:
            if not torch.allclose(node_emb_dict[src], node_emb_src, atol=1e-6):
                print(f"❌ Node {src} có embedding không khớp!")
        else:
            node_emb_dict[src] = node_emb_src  # Lưu embedding đầu tiên

        if tgt in node_emb_dict:
            if not torch.allclose(node_emb_dict[tgt], node_emb_tgt, atol=1e-6):
                print(f"❌ Node {tgt} có embedding không khớp!")
        else:
            node_emb_dict[tgt] = node_emb_tgt  # Lưu embedding đầu tiên

    print("✅ Kiểm tra hoàn tất!")

class CustomedGraphConv(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        aggr: str = 'add',
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_rel = Linear(in_channels[0], out_channels, bias=bias)
        self.lin_root = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_rel.reset_parameters()
        self.lin_root.reset_parameters()


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_weight: OptTensor = None, size: Size = None) -> Tensor:

        if isinstance(x, Tensor):
            x = (x, x)
        assert x[0].size(1) == edge_weight.size(1)
        

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=size)
        out = self.lin_rel(out)

        x_r = x[1]
        if x_r is not None:
            out = out + self.lin_root(x_r)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight * x_j

    def message_and_aggregate(self, adj_t: Adj, x: OptPairTensor) -> Tensor:
        return spmm(adj_t, x[0], reduce=self.aggr)    

class ASTValueEmbedding(nn.Module):
    def __init__(self, pretrained='microsoft/codebert-base', cache_dir="./codebert_cache", embedding_dim=128, batch_size=128):
        super(ASTValueEmbedding, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained, cache_dir=cache_dir)
        self.codebert = AutoModel.from_pretrained(pretrained, cache_dir=cache_dir)

        # Giữ lại chỉ 1 layer Transformer đầu tiên và freeze model
        self.codebert.encoder.layer = self.codebert.encoder.layer[:1]
        for param in self.codebert.parameters():
            param.requires_grad = False

        # Projection layer để giảm dimension
        self.proj = nn.Linear(768, embedding_dim)

        # Batch size để chia nhỏ danh sách input
        self.batch_size = batch_size  

    def forward(self, sentences):
        device = next(self.parameters()).device
        embeddings = []

        # Chia nhỏ sentences thành batch nhỏ để tránh OOM
        for batch in self.chunk_list(sentences, self.batch_size):
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = self.codebert(**inputs)
            batch_embedding = self.proj(outputs.last_hidden_state[:, 0, :])
            embeddings.append(batch_embedding)
        
        return torch.cat(embeddings, dim=0) if embeddings else None

    @staticmethod
    def chunk_list(lst, chunk_size):
        """Chia list thành các phần nhỏ có kích thước chunk_size."""
        for i in range(0, len(lst), chunk_size):
            yield lst[i:i + chunk_size]


class GraphCreator(nn.Module):
    def __init__(self, node_dict: dict, edge_dict: dict, embedding_dim=128):
        super(GraphCreator, self).__init__()
        self.node_dict = node_dict
        self.embedding_dim = embedding_dim
        self.num_nodes = len(node_dict)
        self.num_edge = len(edge_dict)
        self.node_embedding = nn.Embedding(self.num_nodes, embedding_dim)
        self.edge_embedding = nn.Embedding(self.num_edge, embedding_dim)
        self.ast_value_emb = ASTValueEmbedding(embedding_dim=embedding_dim)

    def process_value(self, values: list):
        device = next(self.parameters()).device

        value0_list = [v[0] for v in values]
        value2_list = [v[2] for v in values]

        value0_emb = self.ast_value_emb(value0_list)  # (batch, embed_dim)
        value2_emb = self.ast_value_emb(value2_list)   # (batch, embed_dim)

        value1_tensor = torch.tensor([int(v[1]) for v in values], device=device)
        value1_emb = self.edge_embedding(value1_tensor)  # (batch, embed_dim)

        return torch.stack([value0_emb, value1_emb, value2_emb], dim=1)

    def process_edge(self, edges: list):
        device = next(self.parameters()).device

        node0_list = [e[0] for e in edges]
        node2_list = [e[2] for e in edges]

        node0_tensor = torch.tensor(node0_list, device=device)
        node2_tensor = torch.tensor(node2_list, device=device)

        node0_emb = self.node_embedding(node0_tensor)
        node2_emb = self.node_embedding(node2_tensor)

        edge_tensor = torch.tensor([int(e[1]) for e in edges], device=device)

        edge_emb = self.edge_embedding(edge_tensor)

        return torch.stack([node0_emb, edge_emb, node2_emb], dim=1)
    
    def process_order(self, orders: list):
        device = next(self.parameters()).device

        order0_list = [o[0] for o in orders]
        order2_list = [o[2] for o in orders]

        order0_tensor = torch.tensor(order0_list, device=device)
        order2_tensor = torch.tensor(order2_list, device=device)

        order0_emb = sin_cos_encoding(order0_tensor, self.embedding_dim)
        order2_emb = sin_cos_encoding(order2_tensor, self.embedding_dim)

        edge_tensor = torch.tensor([int(o[1]) for o in orders], device=device)

        edge_emb = self.edge_embedding(edge_tensor)

        return torch.stack([order0_emb, edge_emb, order2_emb], dim=1)

    def get_hetero_data(self, orders: list, node_edge_embedding: torch.tensor):
        device = next(self.parameters()).device
        data = HeteroData()
        
        edge_index = torch.tensor([[e[0], e[2]] for e in orders], dtype=torch.long, device=device).t().contiguous()  # (2, len_cạnh)
        data["node", "edge", "node"].edge_index = edge_index

        edge_emb = node_edge_embedding[:, 1, :]
        data["node", "edge", "node"].edge_attr = edge_emb
        
        node_emb_dict = {}

        for i, (src, tgt) in enumerate(zip(edge_index[0].tolist(), edge_index[1].tolist())):
            if src not in node_emb_dict:  # Lưu một lần duy nhất
                node_emb_dict[src] = node_edge_embedding[i, 0, :]
            if tgt not in node_emb_dict:
                node_emb_dict[tgt] = node_edge_embedding[i, 2, :]
        
        # Chuyển dict thành tensor
        _, node_embs = zip(*node_emb_dict.items())  # Tách keys và values
        # node_tensor = torch.tensor(node_ids, dtype=torch.long, device=device)
        node_emb_tensor = torch.stack(node_embs)  # (num_nodes, embed_dim)
        data["node"].x = node_emb_tensor
        data["node"].num_nodes = len(node_embs)

        return data

    def forward(self, edges: list, orders: list, values: list):
        graph_list = []
        # total_nodes = 0  # Offset tổng số node của các đồ thị trước đó

        for edges, orders, values in zip(edges, orders, values):
            processed_edge = self.process_edge(edges)
            processed_order = self.process_order(orders)
            processed_value = self.process_value(values)
            node_edge_embedding = (processed_edge + processed_order + processed_value) / 3

            graph_data = self.get_hetero_data(orders, node_edge_embedding)

            # Cộng offset vào edge_index
            graph_data["node", "edge", "node"].edge_index
            # total_nodes += graph_data["node"].num_nodes  # Cập nhật tổng số node

            graph_list.append(graph_data)

        return Batch.from_data_list(graph_list)
            
class GCM(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=128, num_layers=4):
        super().__init__()
        self.num_layers = num_layers

        self.gnn_layers = nn.ModuleList([
            HeteroConv({
                ("node", "edge", "node"): CustomedGraphConv(in_dim if i == 0 else hidden_dim, 
                                                            hidden_dim, 
                                                            aggr="mean")
            }, aggr="mean") for i in range(num_layers)
        ])
    
    def cross_graph_attention(self, source_batch: Batch, target_batch: Batch):
        source_batches = source_batch["node"].batch  # (num_nodes,)
        target_batches = target_batch["node"].batch  # (num_nodes,)
        unique_batches = torch.unique(source_batches)

        source_x = source_batch["node"].x  # Lấy node embeddings của source
        target_x = target_batch["node"].x  # Lấy node embeddings của target
        
        new_source_x = source_x.clone()
        new_target_x = target_x.clone() 

        for batch_idx in unique_batches:
            src_mask = source_batches == batch_idx
            tgt_mask = target_batches == batch_idx
            if src_mask.sum() > 0 and tgt_mask.sum() > 0:
                src_nodes = source_x[src_mask]  # (num_src_nodes, hidden_dim)
                tgt_nodes = target_x[tgt_mask]  # (num_tgt_nodes, hidden_dim)

                simmiarity = torch.matmul(src_nodes, tgt_nodes.T)  # (num_src_nodes, num_tgt_nodes)

                # Attention từ source → target (chuẩn hóa theo hàng)
                attention = torch.softmax(simmiarity, dim=1)  # (num_src_nodes, num_tgt_nodes)

                # Lấy embedding mới cho source từ target (vẫn đúng vì tổng hàng = 1)
                new_src_emb = torch.matmul(attention, tgt_nodes)  # (num_src_nodes, hidden_dim)

                # 🚀 Cần chuẩn hóa lại attention để tổng từng cột = 1 trước khi nhân với src_nodes!
                attention_t = torch.softmax(simmiarity, dim=0)  # Chuẩn hóa lại theo cột

                # Lấy embedding mới cho target từ source
                new_tgt_emb = torch.matmul(attention_t.T, src_nodes)  # (num_tgt_nodes, hidden_dim)

                new_source_x.index_copy_(0, src_mask.nonzero(as_tuple=True)[0], new_src_emb)
                new_target_x.index_copy_(0, tgt_mask.nonzero(as_tuple=True)[0], new_tgt_emb)
        
        source_batch["node"].x = new_source_x
        target_batch["node"].x = new_target_x

        return source_batch, target_batch


    def forward(self, source_batch: Batch, target_batch: Batch):
        """
        source_batch, target_batch: Batch của HeteroData chứa nhiều đồ thị
        """
        for i in range(self.num_layers):
            source_x_updated = self.gnn_layers[i](
                x_dict={"node": source_batch["node"].x},
                edge_index_dict={("node", "edge", "node"): source_batch["node", "edge", "node"].edge_index},
                edge_weight_dict={("node", "edge", "node"): source_batch["node", "edge", "node"].edge_attr}
            )["node"].relu()

            target_x_updated = self.gnn_layers[i](
                x_dict={"node": target_batch["node"].x},
                edge_index_dict={("node", "edge", "node"): target_batch["node", "edge", "node"].edge_index},
                edge_weight_dict={("node", "edge", "node"): target_batch["node", "edge", "node"].edge_attr}
            )["node"].relu()

            source_batch["node"].x = source_x_updated.clone()
            target_batch["node"].x = target_x_updated.clone()
            # print(source_batch["node"].x - target_batch["node"].x)

            self.cross_graph_attention(source_batch, target_batch)

        match_scores = []
        unique_batches = torch.unique(source_batch["node"].batch)
        for batch_idx in unique_batches:
            src_mask = source_batch["node"].batch == batch_idx
            tgt_mask = target_batch["node"].batch == batch_idx

            if src_mask.sum() > 0 and tgt_mask.sum() > 0:
                src_nodes = source_batch["node"].x[src_mask].mean(dim=0)
                tgt_nodes = target_batch["node"].x[tgt_mask].mean(dim=0)
                # Tính toán cosine similarity giữa các node embeddings
                match_score = torch.cosine_similarity(src_nodes, tgt_nodes, dim=0)  
                match_scores.append(match_score)
            else:
                match_scores.append(torch.tensor(0.0, device=self.device))

        return torch.stack(match_scores) 
    
if __name__ == '__main__':
    jsonl_dataset = load_from_disk('Processed_BCB_code')
    
    with open("ast_tree.json", "r") as f:
        node_dict = json.load(f)
    with open("ast_edge.json", "r") as f:
        edges_dict = json.load(f)

    graph_creator = GraphCreator(node_dict=node_dict, edge_dict=edges_dict, embedding_dim=128).to(device)
    model = GCM(in_dim=128, hidden_dim=128, num_layers=4).to(device)
    max_order = 0
    max_len = 0
    for i, batch in enumerate(jsonl_dataset.iter(batch_size=2)):
        torch.cuda.empty_cache()
        edges = batch["edges"]
        orders = batch["orders"]
        values = batch["values"]
        with torch.no_grad():
            graph_list = graph_creator(edges, orders, values)
            graph_list2 = graph_creator(edges, orders, values)
            result = model(graph_list, graph_list2)
        print(i, result)
        # break
        