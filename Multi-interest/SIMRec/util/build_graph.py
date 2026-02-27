import torch
import pandas as pd
import numpy as np
import os
import time
import csv
from collections import defaultdict, Counter
from torch_sparse import SparseTensor
from logger import logger


def build_neighbor_common_user_matrix(item_num, graph_name):
    """
    SIMRec
    """
    # graph_name = "music_amazon_item_item_cooccurrence_normalized_train"
    pt_path = f'./graph/{graph_name}_sparse.pt'
    
    logger.info("SIMRec_Graph: " + str(graph_name))

    if os.path.exists(pt_path):
        neighbor_common_user_matrix = torch.load(pt_path)
    else:
        csv_path = f'./graph/{graph_name}.csv'
        df = pd.read_csv(csv_path)
        
        row = torch.tensor(df['item_i'].values, dtype=torch.long)
        col = torch.tensor(df['item_j'].values, dtype=torch.long)
        val = torch.tensor(df['common_people'].values, dtype=torch.float32)
        
        neighbor_common_user_matrix = SparseTensor(row=row, col=col, value=val, sparse_sizes=(item_num, item_num))
        torch.save(neighbor_common_user_matrix, pt_path)

    return neighbor_common_user_matrix


def get_neighbor_common_user_matrix_batch(
    A_sparse,                      # torch_sparse.SparseTensor
    items,                         # Tensor or list/ndarray
    item_num: int,
    type_long: bool = False,
    to_dense: bool = True          # 大表時可考慮 False 保持稀疏
):
    # 取得 A_sparse 的 device（PyG 的 SparseTensor 支援 .device()）
    dev = A_sparse.device()

    # 1) 轉 tensor、放到和 A_sparse 相同的 device，且用 long 當索引
    if torch.is_tensor(items):
        items_t = items.to(device=dev, dtype=torch.long)
    else:
        items_t = torch.as_tensor(items, device=dev, dtype=torch.long)

    # 2) 保證是 (B, L)
    if items_t.dim() == 1:
        items_t = items_t.unsqueeze(1)   # (B, 1)
    B, L = items_t.shape

    # 3) 展平成一個索引向量 (B*L,)
    flattened = items_t.reshape(-1)      # long, same device as A_sparse

    # 4) 用行索引快速擷取 (B*L, item_num)
    #    對 torch_sparse.SparseTensor，row indexing 用 A_sparse[rows]
    result = A_sparse[flattened]         # SparseTensor，大小 (B*L, item_num)

    # 5) 回到 (B, L, item_num)
    if to_dense:
        dense = result.to_dense()        # 可能很大，請評估記憶體
        out = dense.reshape(B, L, item_num)
        if type_long:
            out = out.long()
        return out
    else:
        # 保留為 SparseTensor（如果你後面能吃稀疏）
        # 注意：SparseTensor 沒有直接的 3D 形狀；你可以保留 (B*L, item_num)
        # 或者在外面記錄 B、L 用於之後還原。
        return result, (B, L, item_num)
