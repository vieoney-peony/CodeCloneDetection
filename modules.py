import os

import torch
import torch.nn as nn

import math
import json

from dataset import load_dataset, process_code, process_text, add_edge
from datasets import load_from_disk
from order_flow_ast import JavaASTGraphVisitor, JavaASTLiteralNode, JavaASTBinaryOpNode
from transformers import AutoTokenizer, AutoModel

class GraphNodeEmbedding(nn.Module):
    def __init__(self, node_dict, node_embedding_dim=128, num_nodes=100):
        super(GraphNodeEmbedding, self).__init__()
        self.node_dict = node_dict
        self.node_embedding_dim = node_embedding_dim
        self.num_nodes = num_nodes
        self.embedding = nn.Embedding(num_nodes, node_embedding_dim)

    def forward(self, node_ids, order_ids, value_ids: list):
        return self.embedding(node_ids) 

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
        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
        
        # Forward chỉ qua 1 encoder layer
        outputs = self.codebert(**inputs)
        
        # Lấy vector CLS (token đầu tiên)
        return self.proj(outputs.last_hidden_state[:, 0, :])
        

def sin_cos_encoding(pos, d_model):
    pe = torch.zeros(d_model)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe[0::2] = torch.sin(pos * div_term)
    pe[1::2] = torch.cos(pos * div_term)
    return pe


if __name__ == '__main__':
    jsonl_dataset = load_from_disk('Processed_BCB_dataset')
    
    ast_value_emb = ASTValueEmbedding(embedding_dim=128)

    sentences = [
        "None",
        "+",
        "\n abc, xyz",
        "4708",
        "None",
        "None",
    ]
    res = ast_value_emb(sentences)
    print(res.shape)
    