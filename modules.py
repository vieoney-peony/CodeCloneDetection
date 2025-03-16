import os

import torch
import torch.nn as nn

import math
import json


from dataset import load_dataset, process_code, process_text, add_edge
from datasets import load_from_disk
from order_flow_ast import JavaASTGraphVisitor, JavaASTLiteralNode, JavaASTBinaryOpNode
from transformers import AutoTokenizer, AutoModel

from torch_geometric.data import HeteroData, Batch


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
            


if __name__ == '__main__':
    jsonl_dataset = load_from_disk('Processed_BCB_code')
    
    with open("ast_tree.json", "r") as f:
        node_dict = json.load(f)
    with open("ast_edge.json", "r") as f:
        edges_dict = json.load(f)

    graph_creator = GraphCreator(node_dict=node_dict, edge_dict=edges_dict, embedding_dim=128).to(device)

    max_order = 0
    max_len = 0
    for batch in jsonl_dataset.iter(batch_size=2):
        edges = batch["edges"]
        orders = batch["orders"]
        values = batch["values"]
        
        graph_list = graph_creator(edges, orders, values)
        print(graph_list)
        for x in graph_list["node", "edge", "node"].edge_index:
            max_len = max(max_order, x.max().item())
        print(max_len)
        print(graph_list["node"].ptr)
        print(graph_list["node"].batch)
        break
        