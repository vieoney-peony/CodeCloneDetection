import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import json

from datasets import load_from_disk
from dataset import build_dataset
from transformers import PreTrainedTokenizerFast

from torch_geometric.data import HeteroData, Batch
from torch_geometric.nn import GATConv, GlobalAttention, GENConv

from utils import set_seed
from config import Config

set_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def positional_encoding(max_len, d_model):
    """
    Tạo positional encoding.

    Args:
        max_len: Độ dài tối đa của chuỗi.
        d_model: Kích thước của embedding.

    Returns:
        Một tensor positional encoding có kích thước (max_len, d_model).
    """
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
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
    def __init__(self, tokenizer_path="java_tokenizer.json", embedding_dim=128, max_length=512):
        super(ASTValueEmbedding, self).__init__()

        # Load tokenizer đã train
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
        self.tokenizer.add_special_tokens({
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "cls_token": "<s>",
            "sep_token": "</s>",
            "mask_token": "<mask>",
            "bos_token": "<s>",
            "eos_token": "</s>",
        })
        # Lấy vocab size từ tokenizer
        vocab_size = self.tokenizer.vocab_size
        # print(f"Loaded vocab size: {vocab_size}")
        self.max_length = max_length
        # Tạo nn.Embedding từ vocab
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        
        # self.pos_encoding = positional_encoding(self.max_length, embedding_dim).to(device)

    def to(self, device):
        # self.pos_encoding = self.pos_encoding.to(device)
        return super().to(device)

    def infer_inputs(self, input_ids, attention_mask):
        token_embeddings = self.proj(self.embedding(input_ids))  # (batch_size, seq_len, embedding_dim)
        # token_embeddings = token_embeddings + self.pos_encoding[:input_ids.size(1), :].unsqueeze(0)  # (batch_size, seq_len, embedding_dim)
        token_embeddings = token_embeddings # (batch_size, seq_len, embedding_dim)
        masked_embeddings = token_embeddings * attention_mask.unsqueeze(-1)  # (batch_size, seq_len, embedding_dim)
        sum_embeddings = masked_embeddings.sum(dim=1)  # (batch_size, embedding_dim)
        valid_tokens = attention_mask.sum(dim=1, keepdim=True)  # (batch_size, 1)
        sentence_embeddings = sum_embeddings / valid_tokens.clamp(min=1e-9)  # (batch_size, embedding_dim)

        return sentence_embeddings
    
    def forward(self, sentences):
        device = next(self.parameters()).device
        # Tokenize sentences -> chuyển thành token ID
        encoded_inputs = self.tokenizer(sentences, 
                                        return_tensors="pt", 
                                        padding=True, 
                                        truncation=True, 
                                        max_length=self.max_length)

        input_ids = encoded_inputs["input_ids"].to(device)  # (batch_size, seq_len)
        attention_mask = encoded_inputs["attention_mask"].to(device)  # (batch_size, seq_len)
        sentence_embeddings = self.infer_inputs(input_ids, attention_mask)  # (batch_size, embedding_dim)

        return sentence_embeddings

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

        # order0_emb = sin_cos_encoding(order0_tensor, self.embedding_dim)
        # order2_emb = sin_cos_encoding(order2_tensor, self.embedding_dim)

        order0_emb = torch.zeros((len(order0_list), self.embedding_dim), device=device)
        order2_emb = torch.zeros((len(order2_list), self.embedding_dim), device=device)
        
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
            GENConv(in_dim if i == 0 else hidden_dim, 
                            hidden_dim, 
                            )
            for i in range(num_layers)
        ])
        
        self.cross_attns = nn.ModuleList([
            nn.Sequential(
                nn.MultiheadAttention(hidden_dim, num_heads, batch_first=False),
                nn.LayerNorm(hidden_dim)  # LayerNorm nên đặt ngoài, không trong danh sách []
            ) for _ in range(num_layers)
        ])

        self.mlp_gate = nn.Sequential(nn.Linear(hidden_dim,1),nn.Sigmoid())
        self.pool = GlobalAttention(gate_nn=self.mlp_gate) 
        self.cls_head = nn.Linear(hidden_dim*2, 2) 
    
    def cross_graph_attention(self, source_output, target_output, cross_attn):
        source_x, target_x = source_output['x'], target_output['x']
        source_batches, target_batches = source_output['batch'], target_output['batch']

        attn_layer, norm_layer = cross_attn  

        # Tạo binary mask (src_len, tgt_len), 1 = chặn, 0 = cho phép
        attn_mask = source_batches.view(-1, 1) != target_batches.view(1, -1)  # (src_len, tgt_len)
        attn_mask = attn_mask.to(dtype=torch.bool, device=source_x.device)  # Định dạng chuẩn

        # Thực hiện attention trên toàn bộ graph
        attn_output, _ = attn_layer(source_x.unsqueeze(1), target_x.unsqueeze(1), target_x.unsqueeze(1), attn_mask=attn_mask)
        attn_output_t, _ = attn_layer(target_x.unsqueeze(1), source_x.unsqueeze(1), source_x.unsqueeze(1), attn_mask=attn_mask.T)

        # Chuẩn hóa output
        source_x = norm_layer(attn_output.squeeze(1))
        target_x = norm_layer(attn_output_t.squeeze(1))

        return source_x, target_x


    def forward(self, source_batch: Batch, target_batch: Batch):
        """
        source_batch, target_batch: Batch của HeteroData chứa nhiều đồ thị
        """
        x_source = source_batch["node"].x
        edges_index_source = source_batch["node", "edge", "node"].edge_index
        edges_attr_source = source_batch["node", "edge", "node"].edge_attr
        batch_size_source = source_batch["node"].batch

        x_target = target_batch["node"].x
        edges_index_target = target_batch["node", "edge", "node"].edge_index
        edges_attr_target = target_batch["node", "edge", "node"].edge_attr
        batch_size_target = target_batch["node"].batch

        for i in range(self.num_layers):
            x_source = self.gnn_layers[i](
                x=x_source,
                edge_index=edges_index_source,
                edge_attr=edges_attr_source
            )

            x_target = self.gnn_layers[i](
                x=x_target,
                edge_index=edges_index_target,
                edge_attr=edges_attr_target
            )

            source_output = {"x": x_source, 
                            "edge_index": edges_index_source, 
                            "edge_attr": edges_attr_source, 
                            "batch": batch_size_source}
            target_output = {"x": x_target,
                            "edge_index": edges_index_target, 
                            "edge_attr": edges_attr_target, 
                            "batch": batch_size_target}
            
            x_source, x_target = self.cross_graph_attention(source_output, target_output, self.cross_attns[i])

        pooled_source = self.pool(x_source, batch_size_source)
        pooled_target = self.pool(x_target, batch_size_target)
        out = torch.softmax(self.cls_head(torch.cat([pooled_source, pooled_target], dim=-1)), dim=-1)

        return out

class AttentionFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 1),  
            nn.Softmax(dim=0) 
        )

    def forward(self, outputs):
        # Tính điểm cho mỗi đầu ra
        scores = torch.stack([self.attention_mlp(out).squeeze() for out in outputs]) # (layer, num_node)  
        weights = scores / scores.sum(dim=0, keepdim=True)  # (layer, num_node)
        
        # Weighted sum
        final_output = sum(w.unsqueeze(1) * out for w, out in zip(weights, outputs)) # (num_node, hidden_dim)
        return final_output

class GraphCreatorv2(nn.Module):
    def __init__(self, node_dict: dict, edge_dict: dict, embedding_dim=128, tokenizer_path="java_tokenizer.json"):
        super(GraphCreatorv2, self).__init__()
        self.node_dict = node_dict
        self.embedding_dim = embedding_dim
        self.num_nodes = len(node_dict)
        self.num_edge = len(edge_dict)
        self.node_embedding = nn.Embedding(self.num_nodes, embedding_dim)
        self.edge_embedding = nn.Embedding(self.num_edge, embedding_dim)
        self.ast_value_emb = ASTValueEmbedding(embedding_dim=embedding_dim,
                                            tokenizer_path=tokenizer_path)
    
    def forward(self, batch):
        device = next(self.parameters()).device
        batch = batch.to(device)

        sentence_embeddings = self.ast_value_emb.infer_inputs(batch["node"].input_ids, 
                                                              batch["node"].attention_mask)  
        original_node_emb = self.node_embedding(batch["node"].original_id)  
        batch["node"].x = sentence_embeddings + original_node_emb  # (batch_size, embed_dim)

        edge_emb = self.edge_embedding(batch["node", "edge", "node"].edge_type_id)
        batch["node", "edge", "node"].edge_attr = edge_emb  # (num_edges, embed_dim)

        return batch


class GCMv2(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=128, num_layers=4, num_heads=4):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.gnn_layers = nn.ModuleList([
        GENConv(in_dim if i == 0 else hidden_dim, 
                            hidden_dim, )
            for i in range(num_layers)
        ])
        
        self.cross_attns = nn.ModuleList([
            nn.Sequential(
                nn.MultiheadAttention(hidden_dim, num_heads, batch_first=False),
                nn.LayerNorm(hidden_dim)  # LayerNorm nên đặt ngoài, không trong danh sách []
            ) for _ in range(num_layers)
        ])

        self.layer_fusion = AttentionFusion(hidden_dim)
        self.mlp_gate = nn.Sequential(nn.Linear(hidden_dim,1),nn.Sigmoid())
        self.pool = GlobalAttention(gate_nn=self.mlp_gate) 
        self.cls_head = nn.Linear(hidden_dim*4, 2) 
    
    def cross_graph_attention(self, source_output, target_output, cross_attn):
        source_x, target_x = source_output['x'], target_output['x']
        source_batches, target_batches = source_output['batch'], target_output['batch']

        attn_layer, norm_layer = cross_attn  

        # Tạo binary mask (src_len, tgt_len), 1 = chặn, 0 = cho phép
        attn_mask = source_batches.view(-1, 1) != target_batches.view(1, -1)  # (src_len, tgt_len)
        attn_mask = attn_mask.to(dtype=torch.bool, device=source_x.device)  # Định dạng chuẩn

        # Thực hiện attention trên toàn bộ graph
        attn_output, _ = attn_layer(source_x.unsqueeze(1), target_x.unsqueeze(1), target_x.unsqueeze(1), attn_mask=attn_mask)
        attn_output_t, _ = attn_layer(target_x.unsqueeze(1), source_x.unsqueeze(1), source_x.unsqueeze(1), attn_mask=attn_mask.T)

        # Chuẩn hóa output
        source_x = norm_layer(attn_output.squeeze(1))
        target_x = norm_layer(attn_output_t.squeeze(1))

        return source_x, target_x

    def forward(self, source_batch: Batch, target_batch: Batch):
        """
        source_batch, target_batch: Batch của HeteroData chứa nhiều đồ thị
        """
        # Lấy dữ liệu cho đồ thị nguồn
        x_source = source_batch["node"].x
        edges_index_source = source_batch["node", "edge", "node"].edge_index
        edges_attr_source = source_batch["node", "edge", "node"].edge_attr
        batch_source = source_batch["node"].batch

        # Lấy dữ liệu cho đồ thị đích
        x_target = target_batch["node"].x
        edges_index_target = target_batch["node", "edge", "node"].edge_index
        edges_attr_target = target_batch["node", "edge", "node"].edge_attr
        batch_target = target_batch["node"].batch

        # Khởi tạo danh sách lưu trữ output của mỗi tầng cho nhánh chung và riêng
        common_outputs_source = []
        common_outputs_target = []
        private_outputs_source = []
        private_outputs_target = []

        for i in range(self.num_layers):
            # 1. Áp dụng GNN layer cho từng đồ thị
            x_source = self.gnn_layers[i](
                x=x_source,
                edge_index=edges_index_source,
                edge_attr=edges_attr_source
            )
            x_target = self.gnn_layers[i](
                x=x_target,
                edge_index=edges_index_target,
                edge_attr=edges_attr_target
            )

            # 2. Chuẩn bị input cho cross-attention
            source_output = {"x": x_source, 
                            "edge_index": edges_index_source, 
                            "edge_attr": edges_attr_source, 
                            "batch": batch_source}
            target_output = {"x": x_target,
                            "edge_index": edges_index_target, 
                            "edge_attr": edges_attr_target, 
                            "batch": batch_target}

            # 3. Thực hiện cross-graph attention để trích xuất các đặc trưng chung
            x_source_common, x_target_common = self.cross_graph_attention(source_output, target_output, self.cross_attns[i])
            
            # 4. Tách riêng đặc trưng riêng bằng cách lấy phần hiệu chỉnh
            x_source_private = x_source - x_source_common
            x_target_private = x_target - x_target_common

            # 5. Lưu lại kết quả từ tầng hiện tại
            common_outputs_source.append(x_source_common)
            common_outputs_target.append(x_target_common)
            private_outputs_source.append(x_source_private)
            private_outputs_target.append(x_target_private)

        # 6. Fusion across layers: hợp nhất các đặc trưng từ các tầng (có thể dùng AttentionFusion)
        fused_common_source = self.layer_fusion(common_outputs_source)
        fused_common_target = self.layer_fusion(common_outputs_target)
        fused_private_source = self.layer_fusion(private_outputs_source)
        fused_private_target = self.layer_fusion(private_outputs_target)

        # 7. Global pooling cho từng nhánh
        pooled_common_source = self.pool(fused_common_source, batch_source)
        pooled_common_target = self.pool(fused_common_target, batch_target)
        pooled_private_source = self.pool(fused_private_source, batch_source)
        pooled_private_target = self.pool(fused_private_target, batch_target)

        # 8. Fusion: kết hợp các đặc trưng chung và riêng của mỗi đồ thị
        final_source = torch.cat([pooled_common_source, pooled_private_source], dim=-1)
        final_target = torch.cat([pooled_common_target, pooled_private_target], dim=-1)

        # 9. Cuối cùng, kết hợp hai đồ thị để đưa vào lớp phân loại
        combined = torch.cat([final_source, final_target], dim=-1)
        out = torch.softmax(self.cls_head(combined), dim=-1)

        return out


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

def build_modelv2(config):
    with open(config['node_dict'], "r") as f:
        node_dict = json.load(f)
    with open(config['edge_dict'], "r") as f:
        edges_dict = json.load(f)

    model_config = config['model']
    graph_creator = GraphCreatorv2(node_dict=node_dict, 
                                 edge_dict=edges_dict, 
                                 embedding_dim=model_config['embedding_dim'],
                                 tokenizer_path=config['tokenizer_path'])
    
    model = GCMv2(in_dim=model_config['embedding_dim'], 
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

def inferencev2(graph_creator: GraphCreatorv2, model: GCMv2, graph_source, graph_target):
    graph_source = graph_creator(graph_source)
            
    graph_target = graph_creator(graph_target)
    
    scores = model(graph_source, graph_target)
    return scores

if __name__ == '__main__':
    config = Config("config.yaml")
    jsonl_dataset, txt_dataset, graph_dataset = build_dataset(config)
    graph_creator, model = build_modelv2(config)
    mean_num_node = 0

    for i, batch in enumerate(txt_dataset['train'].iter(batch_size=2)):
        torch.cuda.empty_cache()
        idx1 = batch['idx1']
        idx2 = batch['idx2']
        graph_source = graph_dataset.collate_fn([graph_dataset[idx1[0]]])
        graph_target = graph_dataset.collate_fn([graph_dataset[idx1[0]]])

        scores = inferencev2(graph_creator, model, graph_source, graph_target)
        print("Scores:", scores)
        break
    
    # print(f"Mean number of nodes: {mean_num_node}")
        