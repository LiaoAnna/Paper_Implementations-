import argparse
import torch
import numpy as np
from logger import logger
import random


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', type=str, default='train', help='train | test') # train or test or output
    parser.add_argument('--dataset', type=str, default='book', help='book | taobao') # 数据集
    parser.add_argument('--random_seed', type=int, default=2021)
    parser.add_argument('--hidden_size', type=int, default=64) # 隐藏层维度、嵌入维度
    parser.add_argument('--interest_num', type=int, default=4) # 兴趣的数量
    parser.add_argument('--model_type', type=str, default='MIND', help='DNN | GRU4Rec | MIND | ..') # 模型类型
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning_rate') # 学习率
    parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
    parser.add_argument('--lr_dc_step', type=int, default=30, help='(k), the number of steps after which the learning rate decay')
    parser.add_argument('--max_iter', type=int, default=1000, help='(k)') # 最大迭代次数，单位是k（1000）
    parser.add_argument('--patience', type=int, default=50) # patience，用于early stopping
    parser.add_argument('--topN', type=int, default=50) # default=50
    parser.add_argument('--gpu', type=str, default=None) # None -> cpu
    parser.add_argument('--coef', default=None) # 多样性，用于test
    parser.add_argument('--exp', default='e1')
    parser.add_argument('--add_pos', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--sampled_n', type=int, default=1280)
    parser.add_argument('--sampled_loss', type=str, default='sampled')
    parser.add_argument('--sample_prob', type=int, default=0)
    parser.add_argument('--top_50_result', type=int, default=0)

    # For REMI
    parser.add_argument('--rbeta', type=float, default=0)
    parser.add_argument('--rlambda', type=float, default=0)

    # For graph
    parser.add_argument('--neighbor_graph_name', type=str, default='None')
    parser.add_argument('--SIMRec_graph_name', type=str, default='None')

    # For CL
    parser.add_argument('--cl_alpha', type=float, default=0.25)

    # for MoRec
    parser.add_argument("--num_words_title", type=int, default=30)
    parser.add_argument("--num_words_abstract", type=int, default=50)
    parser.add_argument("--num_words_body", type=int, default=50)
    parser.add_argument("--news_attributes", type=str, default='title')
    parser.add_argument("--bert_model_load", type=str, default='roberta-base')
    parser.add_argument("--freeze_paras_before", type=int, default=165)

    # save item embeddings
    parser.add_argument('--save_item_emb', type=int, default=0)

    # item textual feature
    parser.add_argument("--meta_emb", type=str, default='None')
    parser.add_argument("--meta_emb_2", type=str, default='None')

    # MixRec_ComiRec_SA_B
    parser.add_argument("--mixrec_b", type=float, default=0.5)

    # for test
    parser.add_argument('--best_exp_name', type=str, default=None)

    return parser

def get_dataset_setting(dataset):
    if dataset == 'book':
        path = './data/book_data/'
        item_count = 367982 + 1
        batch_size = 128
        seq_len = 20
        test_iter = 1000
    if dataset == 'bookv':
        path = './data/bookv_data/'
        item_count = 703121 + 1
        batch_size = 128
        seq_len = 20
        test_iter = 1000
    # behaviors:  27158711
    if dataset == 'bookr':
        path = './data/bookr_data/'
        item_count = 1163015 + 1
        batch_size = 128
        seq_len = 20
        test_iter = 1000
    # behaviors:  28723363
    if dataset == 'gowalla':
        path = './data/gowalla_data/'
        item_count = 174605 + 1
        batch_size = 256
        seq_len = 40
        test_iter = 1000
    if dataset == 'gowalla10':
        path = './data/gowalla10_data/'
        item_count = 57445 + 1
        batch_size = 256
        seq_len = 40
        test_iter = 1000
        # behaviors:  2061264
    elif dataset == 'familyTV':
        path = './data/familyTV_data/'
        item_count = 867632 + 1
        batch_size = 256
        seq_len = 30
        test_iter = 1000
    elif dataset == 'kindle':
        path = './data/kindle_data/'
        item_count = 260154 + 1
        batch_size = 128
        seq_len = 20
        test_iter = 200
    elif dataset == 'taobao':
        batch_size = 256
        seq_len = 50
        test_iter = 500
        path = './data/taobao_data/'
        item_count = 1708531
    elif dataset == 'cloth':
        batch_size = 256
        seq_len = 20
        test_iter = 200
        path = './data/cloth_data/'
        item_count = 737822 + 1

    elif dataset == 'tmall':
        batch_size = 256
        seq_len = 100
        test_iter = 200
        path = './data/tmall_data/'
        item_count = 946102 + 1
    elif dataset == 'rocket':
        batch_size = 256
        seq_len = 20
        test_iter = 200
        path = './data/rocket_data/'
        item_count = 81635 + 1
    elif dataset == 'rocket_me':
        batch_size = 256
        seq_len = 20
        test_iter = 200
        path = './data/rocket_data_me/'
        item_count = 90148 + 1
    elif dataset == 'music_amazon':
        batch_size = 256
        seq_len = 20
        test_iter = 200
        path = './data/music_amazon_data/'
        item_count = 10479 + 1
    elif dataset == 'beauty_amazon':
        batch_size = 256
        seq_len = 20
        test_iter = 200
        path = './data/beauty_amazon_data/'
        item_count = 8863 + 1
    elif dataset == 'electronic_amazon':
        batch_size = 256
        seq_len = 20
        test_iter = 200
        path = './data/electronic_amazon_data/'
        item_count = 63002 
    elif dataset == 'movie_amazon':
        batch_size = 256
        seq_len = 20
        test_iter = 200
        path = './data/movie_amazon_data/'
        item_count = 59944 + 1 
    elif dataset == 'cloth_amazon':
        batch_size = 256
        seq_len = 20
        test_iter = 200
        path = './data/cloth_amazon_data/'
        item_count = 376438 + 1 
    return  path, item_count, batch_size, seq_len, test_iter

# def log_args(args):
    
    
#     prob_dic = {
#         0: 'uniform',
#         1: 'log'
#     }
#     logger.info("Param dataset=" + str(args.dataset))
#     logger.info("Param model_type=" + str(args.model_type))
#     logger.info("Param hidden_size=" + str(args.hidden_size))       
#     logger.info("Param dropout=" + str(args.dropout))
#     logger.info("Param layers=" + str(args.layers))
#     logger.info("Param interest_num=" + str(args.interest_num))
#     logger.info("Param add_pos=" + str(args.add_pos == 1))
#     logger.info("Param weight_decay=" + str(args.weight_decay))

#     print("Param dataset=" + str(args.dataset))
#     print("Param model_type=" + str(args.model_type))
#     print("Param hidden_size=" + str(args.hidden_size))
#     print("Param dropout=" + str(args.dropout))
#     print("Param layers=" + str(args.layers))
#     print("Param interest_num=" + str(args.interest_num))
#     print("Param add_pos=" + str(args.add_pos == 1))

#     print("Param weight_decay=" + str(args.weight_decay))


#     logger.info("Param sample_n=" + str(args.sampled_n))
#     logger.info("Param beta=" + str(args.rbeta))
#     logger.info("Param sample_loss=" + str(args.sampled_loss))
#     logger.info("Param sample_prob=" + prob_dic[args.sample_prob])

#     print("Param sampled_n=" + str(args.sampled_n))
#     print("Param beta=" + str(args.rbeta))
#     print("Param sampled_loss=" + str(args.sampled_loss))
#     print("Param sample_prob=" + prob_dic[args.sample_prob])
def log_args(args):
    # 1. 定義特殊值的轉換邏輯 (若有需要轉換顯示方式的參數放這裡)
    prob_dic = {
        0: 'uniform',
        1: 'log'
    }

    # 2. 將 args 轉為字典
    args_dict = vars(args)
    
    # 3. 排序讓 log 更整齊 (可選)
    keys = sorted(args_dict.keys())

    for key in keys:
        val = args_dict[key]     
        # 4. 處理特殊邏輯 (保留你原本代碼中的特殊處理)
        if key == 'sample_prob':
            val = prob_dic.get(val, val) # 如果找不到 key 就回傳原值
        elif key == 'add_pos':
            val = (val == 1) # 轉為布林值      
        # 5. 統一組裝訊息
        log_msg = f"Param {key}={val}"
        
        # 6. 輸出
        logger.info(log_msg)
        print(log_msg)
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
