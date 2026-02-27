from collections import defaultdict
import math
import time
import faiss
import numpy as np
import json
from datetime import datetime
import signal
from util.pytorch_tool import to_tensor
from util.build_graph import get_neighbor_common_user_matrix_batch
import torch

error_flag = {'sig':0}

def sig_handler(signum, frame):
    error_flag['sig'] = signum
    print("segfault core", signum)

signal.signal(signal.SIGSEGV, sig_handler)

# 计算物品多样性，item_list中的所有item两两计算
def compute_diversity(item_list, item_cate_map):
    n = len(item_list)
    diversity = 0.0
    for i in range(n):
        for j in range(i+1, n):
            diversity += item_cate_map[item_list[i]] != item_cate_map[item_list[j]]
    diversity /= ((n-1) * n / 2)
    return diversity

def evaluate(model, test_data, hidden_size, device, k=20, coef=None, item_cate_map=None, args=None, neighbors_matrix=None, neighbor_weights_matrix=None, neighbor_common_user_matrix=None, item_count = None, mode = ""):
    

    topN = k # 评价时选取topN
    if coef is not None:
        coef = float(coef)
    if args.top_50_result and mode == "test": 
        timestamp = datetime.now().strftime("%Y%m%d_%H_%M_%S")
        f = open(f'./top50/{model.name}_{args.dataset}_recommendations_{timestamp}.jsonl', 'w', encoding='utf-8')
        f_target = open(f'./top50/{args.dataset}_target_item.jsonl', 'w', encoding='utf-8')
    else:
        pass

    gpu_indexs = [None]

    # faiss GPU version inner product
    for i in range(1000):
        try:
            if model.name in ["GNN_ComiRec-SA_SIMRec"]:
                item_embs = model.output_items(neighbor_common_user_matrix).cpu().detach().numpy()
            else:
                item_embs = model.output_items().cpu().detach().numpy()
            res = faiss.StandardGpuResources()  # 使用单个GPU
            flat_config = faiss.GpuIndexFlatConfig()
            flat_config.device = device.index

            gpu_indexs[0] = faiss.GpuIndexFlatIP(res, hidden_size, flat_config)  # 建立GPU index用于Inner Product近邻搜索
            gpu_indexs[0].add(item_embs) # 给index添加向量数据
            if error_flag['sig'] == 0:
                break
            else:
                print("core received", error_flag['sig'])
                error_flag['sig'] = 0
        except Exception as e:
            print("error received", e)
        print("Faiss re-try", i)
        time.sleep(5)


    total = 0
    total_recall = 0.0
    total_ndcg = 0.0
    total_hitrate = 0
    total_diversity = 0.0
    
    for _, (users, targets, items, mask, times) in enumerate(test_data): # 一个batch的数据
        
        # 获取用户嵌入
        # 多兴趣模型，shape=(batch_size, num_interest, embedding_dim)
        # 其他模型，shape=(batch_size, embedding_dim)
        time_mat, adj_mat = times
        time_tensor = (to_tensor(time_mat, device), to_tensor(adj_mat, device))

        # get user embeddings
        if  model.name in ["GNN_ComiRec-SA_SIMRec"]:
            neighbors_indices_1hop = neighbors_matrix[items]
            neighbor_weights_1hop = neighbor_weights_matrix[items]
            neighbor_common_user_matrix_batch = get_neighbor_common_user_matrix_batch(neighbor_common_user_matrix, items, item_num=item_count)
            user_embs, _= model(to_tensor(items, device), None, to_tensor(mask, device), time_tensor, device, neighbors_indices_1hop, neighbor_weights_1hop,neighbor_common_user_matrix_batch, train=False)
        else:
            user_embs,_ = model(to_tensor(items, device), None, to_tensor(mask, device), time_tensor, device, train=False)
        user_embs = user_embs.cpu().detach().numpy()
        gpu_index = gpu_indexs[0]

        # 用内积来近邻搜索，实际是内积的值越大，向量越近（越相似）
        if len(user_embs.shape) == 2: # 非多兴趣模型评估
            D, I = gpu_index.search(user_embs, topN) # Inner Product近邻搜索，D为distance，I是index
            for i, (u, iid_list) in enumerate(zip(users, targets)):
            # for i, iid_list in enumerate(targets): # 每个用户的label列表，此处item_id为一个二维list，验证和测试是多label的
                recall = 0
                dcg = 0.0
                item_list = set(I[i]) # I[i]是一个batch中第i个用户的近邻搜索结果，i∈[0, batch_size)
                for no, iid in enumerate(item_list): # 对于每一个label物品
                    if iid in iid_list: # 如果该label物品在近邻搜索的结果中
                        recall += 1
                        dcg += 1.0 / math.log(no+2, 2)
                idcg = 0.0
                for no in range(recall):
                    idcg += 1.0 / math.log(no+2, 2)
                total_recall += recall * 1.0 / len(iid_list)
                if recall > 0: # recall>0当然表示有命中
                    total_ndcg += dcg / idcg
                    total_hitrate += 1
                if coef is not None:
                    total_diversity += compute_diversity(I[i], item_cate_map) # 两个参数分别为推荐物品列表和物品类别字典
                if args.top_50_result and mode == "test": # 保存推荐结果
                    rec_list = I[i].tolist()
                    f.write(json.dumps({
                        "user_id": int(u),
                        "rec_items": rec_list
                    }) + "\n")    
                    f_target.write(json.dumps({
                        "user_id": int(u),
                        "target_items": iid_list
                    }) + "\n")    
        else: # 多兴趣模型评估
            ni = user_embs.shape[1] # num_interest
            user_embs = np.reshape(user_embs, [-1, user_embs.shape[-1]]) # shape=(batch_size*num_interest, embedding_dim)
            D, I = gpu_index.search(user_embs, topN) # Inner Product近邻搜索，D为distance，I是index
            for i, (u, iid_list) in enumerate(zip(users, targets)):
            # for i, iid_list in enumerate(targets): # 每个用户的label列表，此处item_id为一个二维list，验证和测试是多label的
                recall = 0
                dcg = 0.0
                item_list_set = set()
                if coef is None: # 不考虑物品多样性
                    # 将num_interest个兴趣向量的所有topN近邻物品（num_interest*topN个物品）集合起来按照距离重新排序
                    item_list = list(zip(np.reshape(I[i*ni:(i+1)*ni], -1), np.reshape(D[i*ni:(i+1)*ni], -1)))
                    item_list.sort(key=lambda x:x[1], reverse=True) # 降序排序，内积越大，向量越近
                    for j in range(len(item_list)): # 按距离由近到远遍历推荐物品列表，最后选出最近的topN个物品作为最终的推荐物品
                        if item_list[j][0] not in item_list_set and item_list[j][0] != 0:
                            item_list_set.add(item_list[j][0])
                            if len(item_list_set) >= topN:
                                break
                else: # 考虑物品多样性
                    coef = float(coef)
                    # 所有兴趣向量的近邻物品集中起来按距离再次排序
                    origin_item_list = list(zip(np.reshape(I[i*ni:(i+1)*ni], -1), np.reshape(D[i*ni:(i+1)*ni], -1)))
                    origin_item_list.sort(key=lambda x:x[1], reverse=True)
                    item_list = [] # 存放（item_id, distance, item_cate）三元组，要用到物品类别，所以只存放有类别的物品
                    tmp_item_set = set() # 近邻推荐物品中有类别的物品的集合
                    for (x, y) in origin_item_list: # x为索引，y为距离
                        if x not in tmp_item_set and x in item_cate_map:
                            item_list.append((x, y, item_cate_map[x]))
                            tmp_item_set.add(x)
                    cate_dict = defaultdict(int)
                    for j in range(topN): # 选出topN个物品
                        max_index = 0
                        # score = distance - λ * 已选出的物品中与该物品的类别相同的物品的数量（score越大越好）
                        max_score = item_list[0][1] - coef * cate_dict[item_list[0][2]]
                        for k in range(1, len(item_list)): # 遍历所有候选物品，每个循环找出一个score最大的item
                            # 第一次遍历必然先选出第一个物品
                            if item_list[k][1] - coef * cate_dict[item_list[k][2]] > max_score:
                                max_index = k
                                max_score = item_list[k][1] - coef * cate_dict[item_list[k][2]]
                            elif item_list[k][1] < max_score: # 当距离得分小于max_score时，后续物品得分一定小于max_score
                                break
                        item_list_set.add(item_list[max_index][0])
                        # 选出来的物品的类别对应的value加1，这里是为了尽可能选出类别不同的物品
                        cate_dict[item_list[max_index][2]] += 1
                        item_list.pop(max_index) # 候选物品列表中删掉选出来的物品
                if args.top_50_result and mode == "test": # 保存推荐结果
                    # rec_list = list(item_list_set)
                    rec_list = [int(i) for i in item_list_set]
                    f.write(json.dumps({
                        "user_id": int(u),
                        "rec_items": rec_list
                    }) + "\n")
                    f_target.write(json.dumps({
                        "user_id": int(u),
                        "target_items": iid_list
                    }) + "\n")   


                # 上述if-else只是为了用不同方式计算得到最后推荐的结果item列表
                for no, iid in enumerate(item_list_set): # 对于每一个推荐的物品
                    if iid in iid_list: # 如果该推荐的物品在label物品列表中
                        recall += 1
                        dcg += 1.0 / math.log(no+2, 2)
                idcg = 0.0
                for no in range(recall):
                    idcg += 1.0 / math.log(no+2, 2)
                total_recall += recall * 1.0 / len(iid_list) # len(iid_list)表示label数量
                if recall > 0: # recall>0当然表示有命中
                    total_ndcg += dcg / idcg
                    total_hitrate += 1
                if coef is not None:
                    total_diversity += compute_diversity(list(item_list_set), item_cate_map)
        
        total += len(targets) # total增加每个批次的用户数量
    
    recall = total_recall / total # 召回率，每个用户召回率的平均值
    ndcg = total_ndcg / total # NDCG
    hitrate = total_hitrate * 1.0 / total # 命中率
    if coef is None:
        return {'recall': recall, 'ndcg': ndcg, 'hitrate': hitrate}
    diversity = total_diversity * 1.0 / total # 多样性
    return {'recall': recall, 'ndcg': ndcg, 'hitrate': hitrate, 'diversity': diversity}

