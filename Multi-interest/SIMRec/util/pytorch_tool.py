import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from SIMRec import GNN_ComiRec_SA_SIMRec
import pandas as pd
class DataIterator(torch.utils.data.IterableDataset):

    def __init__(self, source,
                 batch_size=128,
                 seq_len=100,
                 train_flag=1,
                 time_span = 128
                ):
        print("Using time span", time_span)
        self.read(source) # 读取数据，获取用户列表和对应的按时间戳排序的物品序列，每个用户对应一个物品list
        self.users = list(self.users) # 用户列表
        self.user_id_all_list = []
        self.hist_item_all_list = []
        self.item_id_all_list = []
        self.time_span = time_span
        self.batch_size = batch_size # 用于训练
        self.eval_batch_size = batch_size # 用于验证、测试
        self.train_flag = train_flag # train_flag=1表示训练
        self.seq_len = seq_len # 历史物品序列的最大长度
        self.index = 0 # 验证和测试时选择用户的位置的标记
        
        print("total user:", len(self.users))
        print("total items:", len(self.items))

    def __iter__(self):
        return self
    
    def output_csv(self, max_steps=None):
        # 讓每次輸出都從乾淨狀態開始
        self.user_id_all_list.clear()
        self.hist_item_all_list.clear()
        self.item_id_all_list.clear()

        if self.train_flag == 1:
            # 訓練是無限流，給一個停止條件
            # 若沒指定 max_steps，就用「遍歷一次所有 user」的步數當預設
            total = len(self.users)
            steps = max_steps or max(1, (total + self.batch_size - 1) // self.batch_size)
            for _ in range(steps):
                _ = self.__next__()
        else:
            # 驗證/測試模式：確保從頭開始
            self.index = 0
            while True:
                try:
                    _ = self.__next__()
                except StopIteration:
                    break

        df = pd.DataFrame({
            'user_id': self.user_id_all_list,
            'hist_items': self.hist_item_all_list,
            'target_item': self.item_id_all_list
        })
        csv_filename = 'train_data.csv' if self.train_flag == 1 else 'test_data.csv'
        df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    # def next(self):
    #     return self.__next__()

    def read(self, source):
        self.graph = {} # key:user_id，value:一个list，放着该user_id所有(item_id,time_stamp)元组，排序后value只保留item_id
        self.time_graph = {}
        self.users = set()
        self.items = set()
        self.times = set()
        with open(source, 'r') as f:
            for line in f:
                conts = line.strip().split(',')
                user_id = int(conts[0])
                item_id = int(conts[1])
                if len(conts) == 3:
                    time_stamp = int(conts[2])
                else:
                    idx = int(conts[2])
                    time_stamp = int(conts[3])
                self.users.add(user_id)
                self.items.add(item_id)
                self.times.add(time_stamp)
                if user_id not in self.graph:
                    self.graph[user_id] = []
                self.graph[user_id].append((item_id, time_stamp))
        for user_id, value in self.graph.items(): # 每个user的物品序列按时间戳排序
            value.sort(key=lambda x: x[1])
            time_list = list(map(lambda x: x[1], value))
            time_min = min(time_list)
            # self.graph[user_id] = list(map(lambda x: [x[0], ], items))
            self.graph[user_id] = [x[0] for x in value] # 排序后只保留了item_id
            # 排序后只保留了item_id, this graph stores the time span of each item in user history (item_timestamp - first_item_timestamp)
            self.time_graph[user_id] = [int(round((x[1] - time_min) / 86400.0) + 1) for x in value] 
        self.users = list(self.users) # 用户列表
        self.items = list(self.items) # 物品列表

    def compute_time_matrix(self, time_seq, item_num):
        time_matrix = np.zeros([self.seq_len, self.seq_len], dtype=np.int32)
        for i in range(item_num):
            for j in range(item_num):
                span = abs(time_seq[i] - time_seq[j])
                if span > self.time_span:
                    time_matrix[i][j] = self.time_span
                else:
                    time_matrix[i][j] = span
        return time_matrix.tolist()

    def compute_adj_matrix(self, mask_seq, item_num):
        node_num = len(mask_seq)

        adj_matrix = np.zeros([node_num, node_num + 2], dtype=np.int32)

        adj_matrix[0][0] = 1
        adj_matrix[0][1] = 1
        adj_matrix[0][-1] = 1

        adj_matrix[item_num - 1][item_num - 1] = 1
        adj_matrix[item_num - 1][item_num] = 1
        adj_matrix[item_num - 1][-1] = 1

        for i in range(1, item_num - 1):
            adj_matrix[i][i] = 1
            adj_matrix[i][i + 1] = 1
            adj_matrix[i][-1] = 1

        if (item_num < node_num):
            for i in range(item_num, node_num):
                adj_matrix[i][0] = 1
                adj_matrix[i][1] = 1
                adj_matrix[i][-1] = 1

        return adj_matrix.tolist()

    def __next__(self):
        '''
        + 選要訓練的user id
        + train dataset + test dataset
            for u in [被選到的user id]:
                + training dataset
                    user的過往紀錄
                    從range(4, len(user的過往紀錄)) 選出 k 
                    選k前面(要輸入到訓練的資料數)個，不足補0
                + test dataset
                    user的過往紀錄後20%
        '''
        if self.train_flag == 1: # 训练
            user_id_list = random.sample(self.users, self.batch_size) # 随机抽取batch_size个user
        else: # 验证、测试，按顺序选取eval_batch_size个user，直到遍历完所有user
            total_user = len(self.users)
            if self.index >= total_user:
                self.index = 0
                raise StopIteration
            user_id_list = self.users[self.index: self.index+self.eval_batch_size]
            self.index += self.eval_batch_size

        item_id_list = []
        hist_time_list = []
        hist_item_list = []
        time_matrix_list = []
        hist_mask_list = []
        adj_matrix_list = []

        for user_id in user_id_list:
            item_list = self.graph[user_id] # 排序后的user的item序列
            time_list = self.time_graph[user_id] # 排序后的user的item序列
            # 这里训练和（验证、测试）采取了不同的数据选取方式
            if self.train_flag == 1: # 训练，选取训练时的label
                k = random.choice(range(4, len(item_list))) # 从[4,len(item_list))中随机选择一个index
                item_id_list.append(item_list[k]) # 该index对应的item加入item_id_list
            else: # 验证、测试，选取该user后20%的item用于验证、测试
                k = int(len(item_list) * 0.8)
                item_id_list.append(item_list[k:])
            # k前的item序列为历史item序列
            if k >= self.seq_len: # 选取seq_len个物品
                hist_item_list.append(item_list[k-self.seq_len: k])
                hist_mask_list.append([1.0] * self.seq_len)
                hist_time_list.append(time_list[k-self.seq_len: k])
                time_matrix_list.append(self.compute_time_matrix(time_list[k - self.seq_len: k], self.seq_len))
                adj_matrix_list.append(self.compute_adj_matrix([1.0] * self.seq_len, self.seq_len))

            else:
                hist_item_list.append(item_list[:k] + [0] * (self.seq_len - k))
                hist_mask_list.append([1.0] * k + [0.0] * (self.seq_len - k))
                hist_time_list.append(time_list[:k] + [0] * (self.seq_len - k))
                time_matrix_list.append(self.compute_time_matrix(time_list[:k] + [0] * (self.seq_len - k), k))
                adj_matrix_list.append(self.compute_adj_matrix([1.0] * k + [0.0] * (self.seq_len - k), k))

        # 返回用户列表（batch_size）、物品列表（label）（batch_size）、
        # 历史物品列表（batch_size，seq_len）、历史物品的mask列表（batch_size，seq_len）
        # time_matrix_list stores the time span of each item in user history (itemi_timestamp - itemj_timestamp)
        self.user_id_all_list += user_id_list
        self.hist_item_all_list += hist_item_list
        self.item_id_all_list += item_id_list
        
        return user_id_list, item_id_list, hist_item_list, hist_mask_list, (time_matrix_list, adj_matrix_list)
def to_tensor(var, device):
    var = torch.Tensor(var)
    var = var.to(device)
    return var.long()

def get_DataLoader(source, batch_size, seq_len, train_flag=1, args=None, item_content=None):
    dataIterator = DataIterator(source, batch_size, seq_len, train_flag)
    dataIterator.output_csv()
    dataLoader_ = DataLoader(dataIterator, batch_size=None, batch_sampler=None, num_workers=0)
    
    return dataLoader_


def get_model(dataset, model_type, item_count, batch_size, hidden_size, interest_num, seq_len, routing_times=3, args=None, device=None, norm_adj = None, bert_model=None):
# def get_model(dataset, model_type, item_count, batch_size, hidden_size, interest_num, seq_len, beta, routing_times=3,):
    add_pos = True
    if args:
        add_pos = args.add_pos == 1
    model_factory = {
        
        "GNN_ComiRec-SA_SIMRec": lambda: GNN_ComiRec_SA_SIMRec(item_count, hidden_size, batch_size, interest_num,
                                                seq_len, add_pos=add_pos, args=args, device=device),
                                                                  
    }

    # 建立模型
    if model_type in model_factory:
        model = model_factory[model_type]()
    else:
        print(f"Unknown model type: {model_type}")
        return
    model.name = model_type
    return model
