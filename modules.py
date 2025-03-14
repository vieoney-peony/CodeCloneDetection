import os

import torch
import torch.nn as nn

import math
import json

from dataset import load_dataset, process_code, process_text, add_edge
from datasets import load_from_disk
from order_flow_ast import JavaASTGraphVisitor, JavaASTLiteralNode, JavaASTBinaryOpNode
from transformers import AutoTokenizer, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class GraphNodeEmbedding(nn.Module):
#     def __init__(self, node_dict, embedding_dim=128, num_nodes=100):
#         super(GraphNodeEmbedding, self).__init__()
#         self.node_dict = node_dict
#         self.embedding_dim = embedding_dim
#         self.num_nodes = num_nodes
#         self.embedding = nn.Embedding(num_nodes, embedding_dim)

#     def forward(self, node_ids):
#         return self.embedding(node_ids) 

class ASTValueEmbedding(nn.Module):
    def __init__(self, pretrained='microsoft/codebert-base', cache_dir = "./codebert_cache", embedding_dim=128):
        super(ASTValueEmbedding, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained, cache_dir=cache_dir)
        self.codebert = AutoModel.from_pretrained(pretrained, cache_dir=cache_dir)
        
        # Giữ lại chỉ 1 layer Transformer đầu tiên
        self.codebert.encoder.layer = self.codebert.encoder.layer[:1]
        
        # Projection layer để giảm dimension
        self.proj = nn.Linear(768, embedding_dim)

    def forward(self, sentences):
        device = next(self.parameters()).device

        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(device)
        
        # Forward chỉ qua 1 encoder layer
        outputs = self.codebert(**inputs)
        
        # Lấy vector CLS (token đầu tiên)
        return self.proj(outputs.last_hidden_state[:, 0, :])

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
        value2_emb = self.ast_value_emb(value2_list)  # (batch, embed_dim)

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

    def forward(self, edges: list, orders: list, values: list):
        processed_edges = []
        processed_orders = []
        processed_values = []
        for edges, orders, values in zip(edges, orders, values):
            processed_edge = self.process_edge(edges)
            processed_order = self.process_order(orders)
            processed_value = self.process_value(values)
            final_edge = torch.mean()
        
        return processed_edges, processed_orders, processed_values
            

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

if __name__ == '__main__':
    jsonl_dataset = load_from_disk('Processed_BCB_code')
    
    with open("ast_tree.json", "r") as f:
        node_dict = json.load(f)
    with open("ast_edge.json", "r") as f:
        edges_dict = json.load(f)

    graph_creator = GraphCreator(node_dict=node_dict, edge_dict=edges_dict, embedding_dim=128).to(device)

    for batch in jsonl_dataset.iter(batch_size=2):
        edges = batch["edges"]
        orders = batch["orders"]
        values = batch["values"]
        
        # Gọi forward của GraphCreator
        processed_edges, processed_orders, processed_values = graph_creator.forward(edges, orders, values)
        print(processed_edges[0].shape)
        print(processed_orders[0].shape)
        print(processed_values[0].shape)
        break