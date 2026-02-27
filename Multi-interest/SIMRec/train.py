from util.build_graph import build_neighbor_common_user_matrix, get_neighbor_common_user_matrix_batch, load_or_generate_neighbors_pt
from util.pytorch_tool import to_tensor, get_DataLoader, get_model
from util.save_file import get_exp_name, save_model, load_model
from evaluated import evaluate
import torch.nn as nn
import torch
import time
import sys
import numpy as np


torch.set_printoptions(
    precision=2,    # 精度，保留小数点后几位，默认4
    threshold=np.inf,
    edgeitems=3,
    linewidth=200,  # 每行最多显示的字符数，默认80，超过则换行显示
    profile=None,
    sci_mode=False  # 用科学技术法显示数据，默认True
)

def train(device, train_file, valid_file, test_file, dataset, model_type, item_count, batch_size, lr, seq_len, 
            hidden_size, interest_num, topN, max_iter, test_iter, decay_step, lr_decay, patience, exp, args):
    # if model_type in ['MIND', 'ComiRec-DR']:
    #     lr = 0.005
    
    
    print("Param lr=" + str(lr))
    logger.info("Param lr=" + str(lr))
    exp_name = get_exp_name(dataset, model_type, batch_size, lr, hidden_size, seq_len, interest_num, topN, exp=exp) # 实验名称
    best_model_path = "best_model/" + exp_name + '/' # 模型保存路径
    
    # prepare data (read data .txt -> DataLoader)
    train_data = get_DataLoader(train_file, batch_size, seq_len, train_flag=1, args=args)
     
    valid_data = get_DataLoader(valid_file, batch_size, seq_len, train_flag=0, args=args)

    test_data = get_DataLoader(test_file, batch_size, seq_len, train_flag=0,  args=args)

    # setting multi-interest model
    model = get_model(dataset, model_type, item_count, batch_size, hidden_size, interest_num, seq_len, args=args, device=device)
    model = model.to(device)
    model.set_device(device)
    # setting sampler
    model.set_sampler(args, device=device)
    # setting loss function & optimizer
    loss_fn = nn.CrossEntropyLoss()
    model.loss_fct = loss_fn

    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=args.weight_decay)
    trials = 0
    # All training time
    start_time = time.time()
    # setting global graph, items' neighbors and weight    
    neighbors_matrix, neighbor_weights_matrix = load_or_generate_neighbors_pt(item_num=item_count, k = 100, graph_name= args.neighbor_graph_name)
    neighbor_common_user_matrix = build_neighbor_common_user_matrix(item_num=item_count, graph_name=args.SIMRec_graph_name)
    neighbor_common_user_matrix = neighbor_common_user_matrix.to(device)
    # check_vector_space_conflict(model, device)
    print("Start training: ")
    # train multi-interest RS
    try:
        total_loss, total_loss_1, total_loss_2, total_loss_3, total_loss_4, total_loss_5  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        iter = 0
        best_metric = 0 # 最佳指标值，在这里是最佳recall值
        #scheduler.step()
        
        for i, (users, targets, items, mask, times) in enumerate(train_data):
            # Interval time
            I_start_time = time.time()
            
            seq_item = to_tensor(items, device)        # (B, L)
            pos_items = to_tensor(targets, device)     # (B,)
            pos_targets = pos_items.unsqueeze(1)       # (B, 1)

            # 這裡要用 seq_item，不要用原始的 list items
            all_items = torch.cat([seq_item, pos_targets], dim=1)  # (B, L+1)
            # set batch item neighbor graph (neighbors_indices_1hop)
            
            neighbor_common_user_matrix_batch = get_neighbor_common_user_matrix_batch(neighbor_common_user_matrix, all_items, item_num=item_count)

            # model train
            model.train()
            iter += 1
            optimizer.zero_grad()
            
            interests, atten, readout, selection = None, None, None, None
            time_mat, adj_mat = times
            times_tensor = (to_tensor(time_mat, device), to_tensor(adj_mat, device))

            
            type_simrec = ["GNN_ComiRec-SA_SIMRec"]
            

            if model_type in type_simrec:
                interests, scores, atten, readout, selection = model(seq_item, pos_items, to_tensor(mask, device), times_tensor, device, neighbor_common_user_matrix_batch)
            else:
                assert False, "model_type not defined"
            # calculate loss

            if model_type in type_simrec:
                loss = model.calculate_optimized_sampled_loss(readout, pos_items, neighbor_common_user_matrix) 
            else:
                assert False, "model_type not defined"
                
            loss.backward()
            optimizer.step()
            total_loss += loss
            
            # evaluate model
            if iter%test_iter == 0:
                model.eval()
                metrics = evaluate(model, valid_data, hidden_size, device, topN, args=args, neighbors_matrix=neighbors_matrix, neighbor_weights_matrix=neighbor_weights_matrix, neighbor_common_user_matrix=neighbor_common_user_matrix, item_count=item_count, mode='valid')
                log_str = 'iter: %d, train loss: %.4f' % (iter, total_loss / test_iter) # 打印loss
                if metrics != {}:
                    log_str += ', ' + ', '.join(['valid ' + key + ': %.6f' % value for key, value in metrics.items()])
                print(exp_name)
                print(log_str)
                logger.info(exp_name)
                logger.info(log_str)

                # 保存recall最佳的模型
                if 'recall' in metrics:
                    recall = metrics['recall']
                    if recall > best_metric:
                        best_metric = recall
                        save_model(model, best_model_path)
                    
                        trials = 0
                    else:
                        trials += 1
                        if trials > patience: # early stopping
                            print("early stopping!")
                            logger.info("early stopping!")
                            break
                
                # 每次test之后loss_sum置零
                total_loss = 0.0
                test_time = time.time()
                print("training time: %.4f min" % ((test_time-start_time)/60.0))
                logger.info("training time: %.4f min" % ((test_time-start_time)/60.0))
                print("time interval: %.4f min" % ((test_time-I_start_time)/60.0))
                logger.info("time interval: %.4f min" % ((test_time-I_start_time)/60.0))
                # logger.info(f"Alpha: {model.cur_alpha:.4f}")
                # print(f"Alpha: {model.cur_alpha:.4f}")
                sys.stdout.flush()

            if iter >= max_iter * 1000: # 超过最大迭代次数，退出训练
                break
            

    except KeyboardInterrupt:
        print('-' * 99)
        print('Exiting from training early')

    load_model(model, best_model_path)
    model.eval()

    # 训练结束后用valid_data测试一次
    metrics = evaluate(model, valid_data, hidden_size, device, topN, args=args, neighbors_matrix=neighbors_matrix, neighbor_weights_matrix=neighbor_weights_matrix, neighbor_common_user_matrix=neighbor_common_user_matrix, item_count=item_count, mode='valid')
    print(', '.join(['Valid ' + key + ': %.6f' % value for key, value in metrics.items()]))
    logger.info(', '.join(['Valid ' + key + ': %.6f' % value for key, value in metrics.items()]))

    # 训练结束后用test_data测试一次
    print("Test result:")
    logger.info("Test result:")
    test_data = get_DataLoader(test_file, batch_size, seq_len, train_flag=0,  args=args)
    
    all_results = []
    for num in [5, 10, 20, 50]:
        metrics = evaluate(model, test_data, hidden_size, device, num, args=args,  neighbors_matrix=neighbors_matrix, neighbor_weights_matrix=neighbor_weights_matrix, neighbor_common_user_matrix=neighbor_common_user_matrix, item_count=item_count, mode='test') 
        for key, value in metrics.items():
            print('test ' + key + f'@{num}' + '=%.6f' % value)
            logger.info('test ' + key + f'@{num}' + '=%.6f' % value)
            all_results.append(value)
    logger.info("All test results: " + ', '.join(['%.6f' % value for value in all_results]))

    # get and save item embeddings
    if args.save_item_emb:
        if model_type in type_simrec_remi:
            item_embeddings =model.output_items(neighbor_common_user_matrix).cpu().detach().numpy()
            np.save(f'./item_emb/{args.output_filename}_item_emb.npy', item_embeddings)


  