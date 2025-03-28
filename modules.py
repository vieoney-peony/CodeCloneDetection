import os
from typing import Tuple, Union

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import math
import json

from datasets import load_from_disk
from order_flow_ast import JavaASTGraphVisitor, JavaASTLiteralNode, JavaASTBinaryOpNode
from transformers import AutoTokenizer, AutoModel

from torch_geometric.data import HeteroData, Batch
from torch_geometric.nn import HeteroConv, MessagePassing, GlobalAttention
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_geometric.utils import spmm

from utils import set_seed
from config import Config

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
        
        edge_index = torch.tensor([[e[0], e[2]] for e in orders], dtype=torch.long, device=device).t().contiguous()
        data["node", "edge", "node"].edge_index = edge_index

        # Gán embedding của cạnh
        data["node", "edge", "node"].edge_attr = node_edge_embedding[:, 1, :]

        # Lấy các node duy nhất và ánh xạ
        unique_nodes = torch.unique(edge_index.flatten())

        # Tạo embedding tensor cho node
        node_embs = torch.zeros((unique_nodes.size(0), node_edge_embedding.size(-1)), device=device)
        
        # print(edge_index[0, :].shape, node_edge_embedding.shape)
        node_embs[edge_index[0, :]] = node_edge_embedding[:, 0, :]  # Source
        node_embs[edge_index[1, :]] = node_edge_embedding[:, 2, :]  # Target

        data["node"].x = node_embs
        data["node"].num_nodes = unique_nodes.size(0)

        return data

    def forward(self, edges_list, orders_list, values_list):
        def process_graph(edges, orders, values):
            node_edge_embedding = (self.process_edge(edges) + 
                                self.process_order(orders) + 
                                self.process_value(values)) / 3
            return self.get_hetero_data(orders, node_edge_embedding)

        graph_list = list(map(process_graph, edges_list, orders_list, values_list))
        return Batch.from_data_list(graph_list)
            
class GCM(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=128, num_layers=4, num_heads=4):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.gnn_layers = nn.ModuleList([
            HeteroConv({
                ("node", "edge", "node"): CustomedGraphConv(in_dim if i == 0 else hidden_dim, 
                                                            hidden_dim, 
                                                            aggr="mean")
            }, aggr="mean") for i in range(num_layers)
        ])
        
        self.cross_attns = nn.ModuleList([
            nn.Sequential(
                nn.MultiheadAttention(hidden_dim, num_heads, batch_first=False),
                nn.LayerNorm(hidden_dim)  # LayerNorm nên đặt ngoài, không trong danh sách []
            ) for _ in range(num_layers)
        ])

        self.mlp_gate=nn.Sequential(nn.Linear(hidden_dim,1),nn.Sigmoid())
        self.pool=GlobalAttention(gate_nn=self.mlp_gate) 

    
    def cross_graph_attention(self, source_batch, target_batch, cross_attn):
        source_batches = source_batch["node"].batch  # (num_source_nodes,)
        target_batches = target_batch["node"].batch  # (num_target_nodes,)

        source_x = source_batch["node"].x  # (num_source_nodes, embed_dim)
        target_x = target_batch["node"].x  # (num_target_nodes, embed_dim)

        attn_layer, norm_layer = cross_attn  

        # Tạo binary mask (src_len, tgt_len), 1 = chặn, 0 = cho phép
        attn_mask = source_batches.view(-1, 1) != target_batches.view(1, -1)  # (src_len, tgt_len)
        attn_mask = attn_mask.to(dtype=torch.bool, device=source_x.device)  # Định dạng chuẩn

        # Thực hiện attention trên toàn bộ graph
        attn_output, _ = attn_layer(source_x.unsqueeze(1), target_x.unsqueeze(1), target_x.unsqueeze(1), attn_mask=attn_mask)
        attn_output_t, _ = attn_layer(target_x.unsqueeze(1), source_x.unsqueeze(1), source_x.unsqueeze(1), attn_mask=attn_mask.T)

        # Chuẩn hóa output
        new_source_x = norm_layer(attn_output.squeeze(1))
        new_target_x = norm_layer(attn_output_t.squeeze(1))

        # Cập nhật giá trị vào batch
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
            )["node"]

            target_x_updated = self.gnn_layers[i](
                x_dict={"node": target_batch["node"].x},
                edge_index_dict={("node", "edge", "node"): target_batch["node", "edge", "node"].edge_index},
                edge_weight_dict={("node", "edge", "node"): target_batch["node", "edge", "node"].edge_attr}
            )["node"]

            source_batch["node"].x = source_x_updated.clone()
            target_batch["node"].x = target_x_updated.clone()
            # print(source_batch["node"].x - target_batch["node"].x)

            self.cross_graph_attention(source_batch, target_batch, self.cross_attns[i])

        # match_scores = []
        # unique_batches = torch.unique(source_batch["node"].batch)
        # for batch_idx in unique_batches:
        #     src_mask = source_batch["node"].batch == batch_idx
        #     tgt_mask = target_batch["node"].batch == batch_idx

            # src_nodes = source_batch["node"].x[src_mask].mean(dim=0)
            # tgt_nodes = target_batch["node"].x[tgt_mask].mean(dim=0)

            # Tính toán dot product giữa các node embeddings
            # match_score = torch.cosine_similarity(src_nodes, tgt_nodes, dim=-1).to(dtype=src_nodes.dtype)
            # match_score = torch.dot(src_nodes, tgt_nodes)
            # match_scores.append(match_score)
        
        pooled_source = self.pool(source_batch["node"].x, source_batch["node"].batch)
        pooled_target = self.pool(target_batch["node"].x, target_batch["node"].batch)

        pooled_source = F.normalize(pooled_source, p=2, dim=-1)
        pooled_target = F.normalize(pooled_target, p=2, dim=-1)
        
        sim = torch.cosine_similarity(pooled_source, pooled_target, dim=-1)
        # return torch.stack(match_scores) 
        
        return sim

def build_model(config):
    with open(config['node_dict'], "r") as f:
        node_dict = json.load(f)
    with open(config['edge_dict'], "r") as f:
        edges_dict = json.load(f)

    model_config = config['model']
    graph_creator = GraphCreator(node_dict=node_dict, 
                                 edge_dict=edges_dict, 
                                 embedding_dim=model_config['embedding_dim'])
    
    model = GCM(in_dim=model_config['embedding_dim'], 
                hidden_dim=model_config['hidden_dim'], 
                num_layers=model_config['num_layers']) 
    
    return graph_creator, model

def inference(graph_creator: GraphCreator, model: GCM, code_batch_source, code_batch_target):
    graph_source = graph_creator(code_batch_source["edges"], 
                                code_batch_source["orders"], 
                                code_batch_source["values"])
            
    graph_target = graph_creator(code_batch_target["edges"], 
                                code_batch_target["orders"], 
                                code_batch_target["values"])
    
    scores = model(graph_source, graph_target)
    return scores

if __name__ == '__main__':
    config = Config("config.yaml")
    jsonl_dataset = load_from_disk('Processed_BCB_code')
    graph_creator, model = build_model(config)
    mean_num_node = 0
    for i, batch in enumerate(jsonl_dataset.iter(batch_size=2)):
        torch.cuda.empty_cache()
        edges = batch["edges"]
        orders = batch["orders"]
        values = batch["values"]
        
        graph_list = graph_creator(edges, orders, values)
        # graph_list2 = graph_creator(edges, orders, values)
        # result = model(graph_list, graph_list2)
        # print(i, result)
        mean_num_node = (mean_num_node*i + graph_list["node"].num_nodes) / (i+1)
        print(mean_num_node)
    
    print(f"Mean number of nodes: {mean_num_node}")
        