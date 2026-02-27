import os
import shutil
import torch


def get_exp_name(dataset, model_type, batch_size, lr, hidden_size, seq_len, interest_num, topN, save=True, exp='e1'):
    extr_name = exp
    para_name = '_'.join([dataset, model_type, 'b'+str(batch_size), 'lr'+str(lr), 'd'+str(hidden_size), 
                            'len'+str(seq_len), 'in'+str(interest_num), 'top'+str(topN)])
    exp_name = para_name + '_' + extr_name

    while os.path.exists('best_model/' + exp_name) and save:
        # flag = input('The exp name already exists. Do you want to cover? (y/n)')
        # if flag == 'y' or flag == 'Y':
        shutil.rmtree('best_model/' + exp_name)
        break
        # else:
        #     extr_name = input('Please input the experiment name: ')
        #     exp_name = para_name + '_' + extr_name

    return exp_name

def save_model(model, Path):
    
    if not os.path.exists(Path):
        os.makedirs(Path)
    torch.save(model.state_dict(), Path + 'model.pt')

def load_model(model, path):
    model.load_state_dict(torch.load(path + 'model.pt'), strict=False)
    print('model loaded from %s' % path)