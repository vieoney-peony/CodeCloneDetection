import os
import random

import javalang
import json

from torch.utils.data import Sampler
from datasets import load_dataset, load_from_disk
from order_flow_ast import JavaASTGraphVisitor, JavaASTLiteralNode, JavaASTBinaryOpNode
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from functools import partial
from utils import set_seed

from config import Config

set_seed(0)

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

    if dataset_config['processed_codes'] is not None:
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
    
    return jsonl_dataset, txt_dataset

if __name__ == '__main__':

    config = Config('config.yaml')
    jsonl_dataset, txt_dataset = build_dataset(config)
    idx_map = {v: i for i, v in enumerate(jsonl_dataset['idx'])}
    print(txt_dataset['train'])
    from torch.utils.data import DataLoader
    trainloader = DataLoader(txt_dataset['test'], batch_size=2, shuffle=True)

    print("Checking...")
    for batch in trainloader:
        for i in range(len(batch['idx1'])):
            if batch['idx1'][i].item() not in idx_map and batch['idx2'][i].item() not in idx_map:
                print(batch['idx1'][i], batch['idx2'][i])
                exit(0)   
        print("OK")