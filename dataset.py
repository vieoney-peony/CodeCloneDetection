import os
import random

import javalang
import copy
import json

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData, Batch
from datasets import load_dataset, load_from_disk
from transformers import PreTrainedTokenizerFast

from order_flow_ast import JavaASTGraphVisitor, JavaASTLiteralNode, JavaASTBinaryOpNode
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from functools import partial
from utils import set_seed

from config import Config

set_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PosNegSampler():
    def __init__(self, dataset):
        self.dataset = dataset  # Lưu dataset gốc
        self.pos_indices = [i for i, x in enumerate(dataset["label"]) if x == 1]
        self.neg_indices = [i for i, x in enumerate(dataset["label"]) if x == 0]

    def sample(self, batch_size):
        pos_sample_ids = random.sample(self.pos_indices, batch_size)  # Lấy batch ngẫu nhiên từ chỉ mục
        neg_sample_ids = random.sample(self.neg_indices, batch_size)

        pos_batch = self.dataset.select(pos_sample_ids)  # Chọn dữ liệu từ chỉ mục
        neg_batch = self.dataset.select(neg_sample_ids)

        return pos_batch, neg_batch
        

def process_text(batch):
    idx1, idx2, label = [], [], []
    
    for text in batch["text"]:
        parts = text.split() 
        idx1.append(int(parts[0]))
        idx2.append(int(parts[1]))
        label.append(int(parts[2]))

    return {"idx1": idx1, "idx2": idx2, "label": label}


def process_code(batch):
    def wrap_code_if_needed(code, idx):
        """Tự động bọc code vào một class nếu cần và kiểm tra cú pháp"""
        code = code.strip()
        wrapped_code = f"public class DummyClass {{\n{code}\n}}"
        # Kiểm tra xem code có parse được không
        try:
            _ = javalang.parse.parse(wrapped_code)
            # method_count = sum(1 for _ in tree.filter(javalang.tree.ClassDeclaration))
            return wrapped_code, code  
        except javalang.parser.JavaSyntaxError:
            print(f"Errorr parse idx={idx}, skip this code.")  
            return None, None 
        

    funcs, idxs = [], []
    for func, idx in zip(batch["func"], batch["idx"]):
        wrapped_func, original_func = wrap_code_if_needed(func, idx)
        if wrapped_func is None:  
            continue  
        funcs.append(wrapped_func)  
        idxs.append(int(idx))

    return {"func": funcs, "idx": idxs}

def get_index_order_value(node_dict:dict, edges_dict:dict, ast_edges: list):
    # print("Processing row:", ast_edges)
    edges = [
        (node_dict[edge[0].label], edges_dict[edge[1]], node_dict[edge[2].label]) 
        for edge in ast_edges
    ]

    orders = [
        (edge[0].sub_index, edges_dict[edge[1]], edge[2].sub_index) 
        for edge in ast_edges
    ]

    values = [
        (
            str(edge[0].value) if isinstance(edge[0], JavaASTLiteralNode) else 
            edge[0].operator if isinstance(edge[0], JavaASTBinaryOpNode) else '"',
            str(edges_dict[edge[1]]),
            str(edge[2].value) if isinstance(edge[2], JavaASTLiteralNode) else 
            edge[2].operator if isinstance(edge[2], JavaASTBinaryOpNode) else '"',
        ) 
        for edge in ast_edges
    ]

    return edges, orders, values

def process_row(row, node_dict, edges_dict):
    code = row["func"]
    ast_tree = javalang.parse.parse(code)
    visitor = JavaASTGraphVisitor()
    visitor.visit(ast_tree)
    return get_index_order_value(node_dict, edges_dict, visitor.edges)

def add_edge(jsonl_dataset, node_dict, edges_dict):
    process_func = partial(process_row, node_dict=node_dict, edges_dict=edges_dict) 
    with ThreadPoolExecutor() as executor:
        result_list= list(tqdm(executor.map(process_func, jsonl_dataset), total=len(jsonl_dataset), desc="Processing ASTs")) # Chạy song song
    
    return result_list  # Trả về danh sách các (edges, orders, values)

def build_dataset(config):
    dataset_config = config['dataset']

    with open(config['node_dict'], "r") as f:
        node_dict = json.load(f)
    with open(config['edge_dict'], "r") as f:
        edges_dict = json.load(f)

    source_codes = os.path.join(dataset_config['dataset'], dataset_config['source_codes'])
    train_path = os.path.join(dataset_config['dataset'], dataset_config['train'])
    valid_path = os.path.join(dataset_config['dataset'], dataset_config['valid'])
    test_path = os.path.join(dataset_config['dataset'], dataset_config['test'])

    txt_dataset = load_dataset('text', data_files={
        'train': train_path,
        'valid': valid_path,
        'test': test_path
        },
        cache_dir=dataset_config['cache_dir']
    )

    txt_dataset = txt_dataset.map(
        process_text, 
        remove_columns=['text'], 
        batched=True, 
        batch_size=100  # Đọc 100 dòng một lần
    )
    
    if dataset_config['processed_codes'] is not None \
            and os.path.exists(dataset_config['processed_codes']):
        jsonl_dataset = load_from_disk(dataset_config['processed_codes'])
    else:
        dataset_config['processed_codes'] = "Processed_BCB_code" # default
        jsonl_dataset = load_dataset('json', 
                                    data_files=source_codes,
                                    split='all',
                                    cache_dir=dataset_config['cache_dir']
                                    )
        jsonl_dataset = jsonl_dataset.map(
            process_code, 
            batched=True, 
            batch_size=100  # Đọc 100 dòng một lần
        )

        results = add_edge(jsonl_dataset, node_dict=node_dict, edges_dict=edges_dict)

        edges_list = [item[0] for item in results]
        orders_list = [item[1] for item in results]
        values_list = [item[2] for item in results]
        jsonl_dataset = jsonl_dataset.add_column("edges", edges_list)
        jsonl_dataset = jsonl_dataset.add_column("orders", orders_list)
        jsonl_dataset = jsonl_dataset.add_column("values", values_list)
        jsonl_dataset.save_to_disk(dataset_config['processed_codes'])

    graph_dataset = GraphDataset(jsonl_dataset, config['tokenizer_path'])

    if dataset_config['processed_graphs'] is not None \
            and os.path.exists(dataset_config['processed_graphs']):
        graph_dataset.load_graph(dataset_config['processed_graphs'])
    else:
        dataset_config['processed_graphs'] = "batched_graphs.pt" # default
        graph_dataset.preprocess()
        torch.save(graph_dataset.graphs, dataset_config['processed_graphs'])

    return jsonl_dataset, txt_dataset, graph_dataset

class GraphDataset(Dataset):
    def __init__(self, jsonl_dataset, tokenizer_path="java_tokenizer.json"):
        self.jsonl_dataset = jsonl_dataset
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
        self.graphs = {}
        self.preprocessed = False

    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx]
    
    def collate_fn(self, batch):
        padded_graphs = []
        # Tìm max_len của toàn batch
        max_len = max([data["node"].input_ids.size(1) for data in batch])

        for data in batch:
            data = data.clone()
            
            input_ids = data["node"].input_ids  # Shape: (num_nodes_i, len_i)
            attention_mask = data["node"].attention_mask  # Shape: (num_nodes_i, len_i)

            # Pad thủ công về max_len
            pad_len = max_len - input_ids.size(1)
            if pad_len > 0:
                input_ids = F.pad(input_ids, (0, pad_len), value=self.tokenizer.pad_token_id)  # Pad bên phải
                attention_mask = F.pad(attention_mask, (0, pad_len), value=0)  # Pad tương ứng

            # Gán lại vào data
            data["node"].input_ids = input_ids
            data["node"].attention_mask = attention_mask
            padded_graphs.append(data)

        batched_graph = Batch.from_data_list(padded_graphs)
        return batched_graph

    def get_str_idx(self, sentences):
        encoded_inputs = self.tokenizer(sentences, 
                                        padding=True, 
                                        truncation=True, 
                                        return_tensors="pt",
                                        max_length=512)
        
        input_ids = encoded_inputs["input_ids"]
        attention_mask = encoded_inputs["attention_mask"]
        return input_ids, attention_mask

    def create_graph(self, edges, orders, values):
        data = HeteroData()
        # id in graph
        edge_index = torch.tensor([[e[0], e[2]] for e in orders], dtype=torch.long).t().contiguous()
        data["node", "edge", "node"].edge_index = edge_index

        # original index
        # edge_original_index = torch.tensor([[e[0], e[2]] for e in edges], dtype=torch.long).t().contiguous()
        # data["node", "edge", "node"].edge_original_index = edge_original_index

        edge_type_id = torch.tensor([e[1] for e in edges], dtype=torch.long)
        data["node", "edge", "node"].edge_type_id = edge_type_id

        unique_nodes = torch.unique(edge_index.flatten())
        # print(f"Unique nodes: {unique_nodes.shape}, {unique_nodes[-10:]}")

        # node features
        node_values = {k: v for node_order, node_value in zip(orders, values) 
                       for k, v in [(node_order[0], node_value[0]), (node_order[2], node_value[2])]} 

        node_features = self.get_str_idx([node_values[int(node_id)] for node_id in unique_nodes])
        
        # node original id
        node_id2original = {k: v for node_order, node_original in zip(orders, edges) 
                       for k, v in [(node_order[0], node_original[0]), (node_order[2], node_original[2])]}
        
        original_id = torch.tensor([v for _, v in sorted(node_id2original.items())])

        # assign to node
        data["node"].input_ids, data["node"].attention_mask = node_features
        data["node"].num_nodes = unique_nodes.size(0)
        data["node"].original_id = original_id

        return data, data["node"].input_ids.size(1)

    def load_graph(self, path):
        if os.path.exists(path):
            self.graphs = torch.load(path, weights_only=False)
            self.preprocessed = True
            print(f"Loaded graphs from {path}")
        else:
            raise FileNotFoundError(f"File {path} not found. Please preprocess the dataset first.")

    def preprocess(self):
        if self.preprocessed:
            print("Dataset already preprocessed.")
            return
        self.preprocessed = True
        max_seq_len = 0
        for i in range(len(self.jsonl_dataset)):
            row = self.jsonl_dataset[i]
            index = row['idx']
            edges = row['edges']
            orders = row['orders']
            values = row['values']
            self.graphs[index], seq_len = self.create_graph(edges, orders, values)
            max_seq_len = max(max_seq_len, seq_len)
            print(f"Processing row {i+1}/{len(self.jsonl_dataset)} with seq_len {seq_len}.")
        
        print(f"Max sequence length: {max_seq_len}")

if __name__ == '__main__':
    config = Config('config.yaml')
    jsonl_dataset, txt_dataset, graph_dataset = build_dataset(config)
    
    # dataloader = torch.utils.data.DataLoader(
    #     graph_dataset, 
    #     batch_size=4, 
    #     shuffle=False, 
    #     collate_fn=graph_dataset.collate_fn
    # )

    # for batch in dataloader:
    #     print(batch)
    #     break