import javalang
from datasets import load_dataset
from order_flow_ast import JavaASTGraphVisitor, JavaASTLiteralNode, JavaASTBinaryOpNode
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from functools import partial
import json

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
            edge[0].operator if isinstance(edge[0], JavaASTBinaryOpNode) else 'None',
            edges_dict[edge[1]],
            str(edge[2].value) if isinstance(edge[2], JavaASTLiteralNode) else 
            edge[2].operator if isinstance(edge[2], JavaASTBinaryOpNode) else 'None',
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
    print("Results:", edges_list)