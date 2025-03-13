import torch
import torch.nn as nn

import math
import json

import javalang
from dataset import load_dataset, process_code, process_text, add_edge
from order_flow_ast import JavaASTGraphVisitor, JavaASTLiteralNode, JavaASTBinaryOpNode

class GraphNodeEmbedding(nn.Module):
    def __init__(self, node_dict, node_embedding_dim=128, num_nodes=100):
        super(GraphNodeEmbedding, self).__init__()
        self.node_dict = node_dict
        self.node_embedding_dim = node_embedding_dim
        self.num_nodes = num_nodes
        self.embedding = nn.Embedding(num_nodes, node_embedding_dim)

    def get_index_order_value(self, ast_edges: list):
        edges = [
            (self.node_dict[edge[0].label], edge[1], self.node_dict[edge[2].label]) 
            for edge in ast_edges
        ]

        orders = [
            (edge[0].sub_index, edge[1], edge[2].sub_index) 
            for edge in ast_edges
        ]

        values = [
            (
                str(edge[0].value) if isinstance(edge[0], JavaASTLiteralNode) else 
                edge[0].operator if isinstance(edge[0], JavaASTBinaryOpNode) else 'None',
                edge[1],
                str(edge[2].value) if isinstance(edge[2], JavaASTLiteralNode) else 
                edge[2].operator if isinstance(edge[2], JavaASTBinaryOpNode) else 'None',
            ) 
            for edge in ast_edges
        ]

        return edges, orders, values

    def forward(self, node_ids, order_ids, value_ids: list):
        return self.embedding(node_ids) * math.sqrt(self.d_model)


def sin_cos_encoding(pos, d_model):
    pe = torch.zeros(d_model)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe[0::2] = torch.sin(pos * div_term)
    pe[1::2] = torch.cos(pos * div_term)
    return pe




if __name__ == '__main__':
    jsonl_dataset = load_dataset('json', data_files='BCB_dataset/data.jsonl', 
                                 split='all', 
                                 cache_dir="./BCB_cache")

    txt_dataset = load_dataset('text', data_files={
        'train': 'BCB_dataset/train.txt',
        'valid': 'BCB_dataset/valid.txt',
        'test': 'BCB_dataset/test.txt'},
        cache_dir="./BCB_cache"
    )

    jsonl_dataset = jsonl_dataset.map(
        process_code, 
        batched=True, 
        batch_size=100  # Đọc 100 dòng một lần
    )

    txt_dataset = txt_dataset.map(
        process_text, 
        remove_columns=['text'], 
        batched=True, 
        batch_size=100  # Đọc 100 dòng một lần
    )


    print("JSONL Data:", jsonl_dataset)
    print("TXT Data:", txt_dataset)

    # df_jsonl = jsonl_dataset.to_pandas()

    print("Sampling data:")
    # for record in tqdm(jsonl_dataset, desc="Processing records"):
    #     code = record["func"]
    #     ast_tree = javalang.parse.parse(code)
    #     visitor = JavaASTGraphVisitor()
    #     visitor.visit(ast_tree)

    with open("ast_tree.json", "r") as f:
        node_dict = json.load(f)
    with open("ast_edge.json", "r") as f:
        edges_dict = json.load(f)
    results = add_edge(jsonl_dataset, node_dict=node_dict, edges_dict=edges_dict)
    edges_list = [item[0] for item in results]
    jsonl_dataset = jsonl_dataset.add_column('edges', edges_list)
    # jsonl_dataset.add_column('orders', results[1])
    # jsonl_dataset.add_column('values', results[2])
    print(jsonl_dataset)
    