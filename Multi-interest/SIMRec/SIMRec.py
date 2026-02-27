import torch
import torch.nn as nn
import torch.nn.functional as F
from math import isclose
import math
from torch.nn.init import xavier_uniform_, xavier_normal_
import numpy as np
from torch_sparse import SparseTensor

BACKOFF_PROB = 1e-10

class AliasMultinomial(torch.nn.Module):
    '''Alias sampling method to speedup multinomial sampling
    The alias method treats multinomial sampling as a combination of uniform sampling and
    bernoulli sampling. It achieves significant acceleration when repeatedly sampling from
    the save multinomial distribution.
    Attributes:
        - probs: the probability density of desired multinomial distribution
    Refs:
        - https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    '''
    def __init__(self, probs):
        super(AliasMultinomial, self).__init__()

        # @todo calculate divergence
        assert abs(probs.sum().item() - 1) < 1e-5, 'The noise distribution must sum to 1'

        cpu_probs = probs.cpu()
        K = len(probs)

        # such a name helps to avoid the namespace check for nn.Module
        self_prob = [0] * K
        self_alias = [0] * K

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger = []
        for idx, prob in enumerate(cpu_probs):
            self_prob[idx] = K*prob
            if self_prob[idx] < 1.0:
                smaller.append(idx)
            else:
                larger.append(idx)

        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self_alias[small] = large
            self_prob[large] = (self_prob[large] - 1.0) + self_prob[small]

            if self_prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller+larger:
            self_prob[last_one] = 1

        self.register_buffer('prob', torch.Tensor(self_prob))
        self.register_buffer('alias', torch.LongTensor(self_alias))

    def draw(self, *size):
        """Draw N samples from multinomial
        Args:
            - size: the output size of samples
        """
        max_value = self.alias.size(0)

        kk = self.alias.new(*size).random_(0, max_value).long().view(-1)
        prob = self.prob[kk]
        alias = self.alias[kk]
        # b is whether a random number is greater than q
        b = torch.bernoulli(prob).long()
        oq = kk.mul(b)
        oj = alias.mul(1 - b)

        return (oq + oj).view(size)
    
class NCELoss(nn.Module):
    """Noise Contrastive Estimation
    NCE is to eliminate the computational cost of softmax
    normalization.
    There are 3 loss modes in this NCELoss module:
        - nce: enable the NCE approximation
        - sampled: enabled sampled softmax approximation
        - full: use the original cross entropy as default loss
    They can be switched by directly setting `nce.loss_type = 'nce'`.
    Ref:
        X.Chen etal Recurrent neural network language
        model training with noise contrastive estimation
        for speech recognition
        https://core.ac.uk/download/pdf/42338485.pdf
    Attributes:
        noise: the distribution of noise
        noise_ratio: $\frac{#noises}{#real data samples}$ (k in paper)
        norm_term: the normalization term (lnZ in paper), can be heuristically
        determined by the number of classes, plz refer to the code.
        reduction: reduce methods, same with pytorch's loss framework, 'none',
        'elementwise_mean' and 'sum' are supported.
        loss_type: loss type of this module, currently 'full', 'sampled', 'nce'
        are supported
    Shape:
        - noise: :math:`(V)` where `V = vocabulary size`
        - target: :math:`(B, N)`
        - loss: a scalar loss by default, :math:`(B, N)` if `reduction='none'`
    Input:
        target: the supervised training label.
        args&kwargs: extra arguments passed to underlying index module
    Return:
        loss: if `reduction='sum' or 'elementwise_mean'` the scalar NCELoss ready for backward,
        else the loss matrix for every individual targets.
    """

    def __init__(self,
                 noise,
                 noise_ratio=100,
                 norm_term='auto',
                 reduction='elementwise_mean',
                 per_word=False,
                 loss_type='nce',
                 beta = 0,
                 device=None
                 ):
        super(NCELoss, self).__init__()
        self.device = device
        # Re-norm the given noise frequency list and compensate words with
        # extremely low prob for numeric stability
        self.update_noise(noise)

        # @todo quick path to 'full' mode
        # @todo if noise_ratio is 1, use all items as samples
        self.noise_ratio = noise_ratio
        self.beta = beta
        if norm_term == 'auto':
            self.norm_term = math.log(noise.numel())
        else:
            self.norm_term = norm_term
        self.reduction = reduction
        self.per_word = per_word
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.loss_type = loss_type

    def update_noise(self, noise):
        probs = noise / noise.sum()
        probs = probs.clamp(min=BACKOFF_PROB)
        renormed_probs = probs / probs.sum()
        # import pdb; pdb.set_trace()
        self.register_buffer('logprob_noise', renormed_probs.log())
        self.alias = AliasMultinomial(renormed_probs)

    def forward(self, target, input, embs, interests=None, loss_fn = None, *args, **kwargs):
        """compute the loss with output and the desired target
        The `forward` is the same among all NCELoss submodules, it
        takes care of generating noises and calculating the loss
        given target and noise scores.
        """

        batch = target.size(0)
        max_len = target.size(1)
        if self.loss_type != 'full':

            # use all or sampled
            # noise_samples = self.get_noise(batch, max_len)
            noise_samples = torch.arange(embs.size(0)).to(self.device).unsqueeze(0).unsqueeze(0).repeat(batch, 1, 1) if self.noise_ratio == 1 else self.get_noise(batch, max_len)

            logit_noise_in_noise = self.logprob_noise[noise_samples.data.view(-1)].view_as(noise_samples)
            logit_target_in_noise = self.logprob_noise[target.data.view(-1)].view_as(target)

            # B,N,Nr

            # (B,N), (B,N,Nr)
            logit_noise_in_noise = self.logprob_noise[noise_samples.data.view(-1)].view_as(noise_samples)
            logit_target_in_noise = self.logprob_noise[target.data.view(-1)].view_as(target)

            logit_target_in_model, logit_noise_in_model = self._get_logit(target, noise_samples, input, embs, *args, **kwargs)



            if self.loss_type == 'nce':
                if self.training:
                    loss = self.nce_loss(
                        logit_target_in_model, logit_noise_in_model,
                        logit_noise_in_noise, logit_target_in_noise,
                    )
                else:
                    # directly output the approximated posterior
                    loss = - logit_target_in_model
            elif self.loss_type == 'sampled':
                loss = self.sampled_softmax_loss(
                    logit_target_in_model, logit_noise_in_model,
                    logit_noise_in_noise, logit_target_in_noise,
                )
            # NOTE: The mix mode is still under investigation
            elif self.loss_type == 'mix' and self.training:
                loss = 0.5 * self.nce_loss(
                    logit_target_in_model, logit_noise_in_model,
                    logit_noise_in_noise, logit_target_in_noise,
                )
                loss += 0.5 * self.sampled_softmax_loss(
                    logit_target_in_model, logit_noise_in_model,
                    logit_noise_in_noise, logit_target_in_noise,
                )

            else:
                current_stage = 'training' if self.training else 'inference'
                raise NotImplementedError(
                    'loss type {} not implemented at {}'.format(
                        self.loss_type, current_stage
                    )
                )

        else:
            # Fallback into conventional cross entropy
            loss = self.ce_loss(target, *args, **kwargs)

        if self.reduction == 'elementwise_mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def get_noise(self, batch_size, max_len):
        """Generate noise samples from noise distribution"""

        noise_size = (batch_size, max_len, self.noise_ratio)
        if self.per_word:
            noise_samples = self.alias.draw(*noise_size)
        else:
            noise_samples = self.alias.draw(1, 1, self.noise_ratio).expand(*noise_size)

        noise_samples = noise_samples.contiguous()
        return noise_samples

    def _get_logit(self, target_idx, noise_idx,input, embs, *args, **kwargs):
        """Get the logits of NCE estimated probability for target and noise
        Both NCE and sampled softmax Loss are unchanged when the probabilities are scaled
        evenly, here we subtract the maximum value as in softmax, for numeric stability.
        Shape:
            - Target_idx: :math:`(N)`
            - Noise_idx: :math:`(N, N_r)` where `N_r = noise ratio`
        """

        target_logit, noise_logit = self.get_score(target_idx, noise_idx, input, embs, *args, **kwargs)

        # import pdb; pdb.set_trace()
        target_logit = target_logit.sub(self.norm_term)
        noise_logit = noise_logit.sub(self.norm_term)
        # import pdb; pdb.set_trace()
        return target_logit, noise_logit

    def get_score(self, target_idx, noise_idx, input, embs, *args, **kwargs):
        """Get the target and noise score
        Usually logits are used as score.
        This method should be override by inherit classes
        Returns:
            - target_score: real valued score for each target index
            - noise_score: real valued score for each noise index
        """
        original_size = target_idx.size()

        # flatten the following matrix
        input = input.contiguous().view(-1, input.size(-1))
        target_idx = target_idx.view(-1)
        noise_idx = noise_idx[0, 0].view(-1)
        # import pdb; pdb.set_trace()
        target_batch = embs[target_idx]
        # import pdb; pdb.set_trace()
        # target_bias = self.bias.index_select(0, target_idx)  # N
        target_score = torch.sum(input * target_batch, dim=1) # N X E * N X E

        noise_batch = embs[noise_idx]  # Nr X H
        noise_score = torch.matmul(
            input, noise_batch.t()
        )
        return target_score.view(original_size), noise_score.view(*original_size, -1)

    def ce_loss(self, target_idx, *args, **kwargs):
        """Get the conventional CrossEntropyLoss
        The returned loss should be of the same size of `target`
        Args:
            - target_idx: batched target index
            - args, kwargs: any arbitrary input if needed by sub-class
        Returns:
            - loss: the estimated loss for each target
        """
        raise NotImplementedError()

    def nce_loss(self, logit_target_in_model, logit_noise_in_model, logit_noise_in_noise, logit_target_in_noise):
        """Compute the classification loss given all four probabilities
        Args:
            - logit_target_in_model: logit of target words given by the model (RNN)
            - logit_noise_in_model: logit of noise words given by the model
            - logit_noise_in_noise: logit of noise words given by the noise distribution
            - logit_target_in_noise: logit of target words given by the noise distribution
        Returns:
            - loss: a mis-classification loss for every single case
        """

        # NOTE: prob <= 1 is not guaranteed
        logit_model = torch.cat([logit_target_in_model.unsqueeze(2), logit_noise_in_model], dim=2)
        logit_noise = torch.cat([logit_target_in_noise.unsqueeze(2), logit_noise_in_noise], dim=2)

        # predicted probability of the word comes from true data distribution
        # The posterior can be computed as following
        # p_true = logit_model.exp() / (logit_model.exp() + self.noise_ratio * logit_noise.exp())
        # For numeric stability we compute the logits of true label and
        # directly use bce_with_logits.
        # Ref https://pytorch.org/docs/stable/nn.html?highlight=bce#torch.nn.BCEWithLogitsLoss
        logit_true = logit_model - logit_noise - math.log(self.noise_ratio)

        label = torch.zeros_like(logit_model)
        label[:, :, 0] = 1

        loss = self.bce_with_logits(logit_true, label).sum(dim=2)
        return loss

    def sampled_softmax_loss(self, logit_target_in_model, logit_noise_in_model, logit_noise_in_noise, logit_target_in_noise):
        """Compute the sampled softmax loss based on the tensorflow's impl"""
        ori_logits = torch.cat([logit_target_in_model.unsqueeze(2), logit_noise_in_model], dim=2)
        q_logits = torch.cat([logit_target_in_noise.unsqueeze(2), logit_noise_in_noise], dim=2)

        # subtract Q for correction of biased sampling
        logits = ori_logits - q_logits
        labels = torch.zeros_like(logits.narrow(2, 0, 1)).squeeze(2).long()

        if self.beta == 0:
            loss = self.ce(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            ).view_as(labels)

        if self.beta != 0:
            x = ori_logits.view(-1, ori_logits.size(-1))
            x = x - torch.max(x, dim = -1)[0].unsqueeze(-1)
            pos = torch.exp(x[:,0])
            neg = torch.exp(x[:,1:])
            imp = (self.beta * x[:,1:] -  torch.max(self.beta * x[:,1:],dim = -1)[0].unsqueeze(-1)).exp()
            reweight_neg = (imp*neg).sum(dim = -1) / imp.mean(dim = -1)
            if torch.isinf(reweight_neg).any() or torch.isnan(reweight_neg).any():
                import pdb; pdb.set_trace()
            Ng = reweight_neg

            stable_logsoftmax = -(x[:,0] - torch.log(pos + Ng))
            loss = torch.unsqueeze(stable_logsoftmax, 1)

        return loss
    
def build_uniform_noise(number):
    total = number
    freq = torch.Tensor([1.0] * number).cuda()
    noise = freq / total 
    assert abs(noise.sum() - 1) < 0.001
    return noise

def build_log_noise(number):
    total = number
    freq = torch.Tensor([1.0] * number).cuda()
    noise = freq / total
    for i in range(number):
        noise[i] = (np.log(i + 2) - np.log(i + 1)) / np.log(number + 1)

    assert abs(noise.sum() - 1) < 0.001
    return noise

def build_noise(number, args=None):
    if args.sample_prob == 0:
        return build_uniform_noise(number)
    if args.sample_prob == 1:
        return build_log_noise(number)


class GNN_ComiRec_SA_SIMRec(nn.Module):
    def __init__(self, item_num, hidden_size, batch_size, interest_num=4, seq_len=50, add_pos=True, beta=0, args=None, device=None):
        super(GNN_ComiRec_SA_SIMRec, self).__init__()
        self.name = 'GNN_ComiRec_SA_SIMRec'
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.item_num = item_num
        self.seq_len = seq_len
        self.beta = beta
        self.embeddings = nn.Embedding(self.item_num, self.hidden_size, padding_idx=0)
        self.interest_num = interest_num
        self.num_heads = interest_num
        # self.interest_num = interest_num
        self.hard_readout = True
        self.add_pos = add_pos
        if self.add_pos:
            self.position_embedding = nn.Parameter(torch.Tensor(1, self.seq_len, self.hidden_size))
        self.linear1 = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size * 4, bias=False),
                nn.Tanh()
            )
        self.linear2 = nn.Linear(self.hidden_size * 4, self.num_heads, bias=False)
        self.reset_parameters()

    def set_device(self, device):
        self.device = device

    def set_sampler(self, args, device=None):
        self.is_sampler = True
        if args.sampled_n == 0:
            self.is_sampler = False
            return

        self.sampled_n = args.sampled_n

        noise = build_noise(self.item_num, args)

        self.sample_loss = NCELoss(noise=noise,
                                       noise_ratio=self.sampled_n,
                                       norm_term=0,
                                       reduction='elementwise_mean',
                                       per_word=False,
                                       loss_type=args.sampled_loss,
                                       beta=self.beta,
                                       device=device
                                       )

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)

    def reset_parameters(self, initializer=None):
        for weight in self.parameters():
            torch.nn.init.kaiming_normal_(weight)


    def read_out(self, user_eb, label_eb):

        # 这个模型训练过程中label是可见的，此处的item_eb就是label物品的嵌入
        atten = torch.matmul(user_eb, # shape=(batch_size, interest_num, hidden_size)
                        torch.reshape(label_eb, (-1, self.hidden_size, 1)) # shape=(batch_size, hidden_size, 1)
                        ) # shape=(batch_size, interest_num, 1)

        atten = F.softmax(torch.pow(torch.reshape(atten, (-1, self.interest_num)), 1), dim=-1) # shape=(batch_size, interest_num)

        if self.hard_readout: # 选取interest_num个兴趣胶囊中的一个，MIND和ComiRec都是用的这种方式
           
            readout = torch.reshape(user_eb, (-1, self.hidden_size))[
                        (torch.argmax(atten, dim=-1) + torch.arange(label_eb.shape[0], device=user_eb.device) * self.interest_num).long()] # shape=(batch_size, hidden_size)
        else: # 综合interest_num个兴趣胶囊，论文及代码实现中没有使用这种方法
            readout = torch.matmul(torch.reshape(atten, (label_eb.shape[0], 1, self.interest_num)), # shape=(batch_size, 1, interest_num)
                                user_eb # shape=(batch_size, interest_num, hidden_size)
                                ) # shape=(batch_size, 1, hidden_size)
            readout = torch.reshape(readout, (label_eb.shape[0], self.hidden_size)) # shape=(batch_size, hidden_size)
        # readout是vu堆叠成的矩阵（一个batch的vu）（vu可以说就是最终的用户嵌入）
        # selection recoreds select the index of the most relevant interest capsule 
        selection = torch.argmax(atten, dim=-1)
        return readout, selection
    

    # def calculate_score(self, user_eb):
    #     all_items = self.embeddings.weight
    #     scores = torch.matmul(user_eb, all_items.transpose(1, 0)) # [b, n]
    #     return scores

    def calculate_full_loss(self, loss_fn, scores, target, interests):
        return loss_fn(scores, target)


    # def calculate_sampled_loss(self, readout, pos_items, selection, interests):
    #     return self.sample_loss(pos_items.unsqueeze(-1), readout, self.embeddings.weight) 
    def calculate_sampled_loss(self, readout, pos_items, selection, interests, neighbor_common_user_matrix):
        
        # 1. 根據論文 Eq. (17) 獲取增強後的物品嵌入矩陣 E_I
        # 注意：output_items 已經實現了 A * E_tilde 的邏輯
        enhanced_item_embedding = self.output_items(neighbor_common_user_matrix)

        # 2. 使用增強後的嵌入矩陣計算 Loss
        # 這樣正樣本和負樣本都會從 E_I 中採樣，而不是從原始的 E_tilde 中採樣
        return self.sample_loss(pos_items.unsqueeze(-1), readout, enhanced_item_embedding)   
    
    
    def forward(self, item_list, label_list, mask, times, device,  neighbor_common_user_matrix=None, train=True):
        # item_eb = self.embeddings(item_list)
        # print(f"first item_eb shape: {item_eb.shape}, device: {item_eb.device}")
        
        # Graph info
        batch_size, L = item_list.shape
        neighbor_common_user_matrix = neighbor_common_user_matrix.to(self.embeddings.weight.device)
        # print(f"neighbor_common_user_matrix shape: {neighbor_common_user_matrix.shape}, device: {neighbor_common_user_matrix.device}")
        # print(f"self.embeddings.weight shape: {self.embeddings.weight.shape}, device: {self.embeddings.weight.device}")
        new_embedding = torch.matmul(neighbor_common_user_matrix, self.embeddings.weight)
        # new_embedding_layer = nn.Embedding.from_pretrained(new_embedding, freeze=False)
        # print(f"new_embedding shape: {new_embedding.shape}, device: {new_embedding.device}")
        
        # Item embeddings
        X = self.embeddings(item_list)  # (batch_size, L, d)
        
        if train:
            item_eb =  new_embedding[:, :-1, :] 
        else:
            item_eb = new_embedding
        
        item_eb = item_eb * torch.reshape(mask, (-1, self.seq_len, 1))
        if train:
            # label_eb : positive samples embedding
            label_eb =  new_embedding[:, -1, :] 
        # 用户多兴趣向量
        user_eb, item_att_w = self.multi_interest_forward(item_eb, mask)


        if not train:
            return user_eb, None
        # readout: the final user embedding(select the most important interest)
        # selection: the index of the most important interest
        readout, selection = self.read_out(user_eb, label_eb)
        # user_eb*item_eb, all the items from dataset
        scores = None if self.is_sampler else self.calculate_score(readout)
        return user_eb, scores, item_att_w, readout, selection
    
    def output_items(self, neighbor_common_user_matrix=None):
        """
        獲取所有物品的嵌入向量。
        執行 Eq. (17): E_I = A * E_tilde
        """
        # if neighbor_common_user_matrix is None:
        #     return self.embeddings.weight

        # 1. 確保設備一致
        device = self.embeddings.weight.device
        if neighbor_common_user_matrix.device != device:
            neighbor_common_user_matrix = neighbor_common_user_matrix.to(device)  
        return neighbor_common_user_matrix.matmul(self.embeddings.weight)
    
    def multi_interest_forward(self, item_eb, mask):
         # 历史物品嵌入序列，shape=(batch_size, maxlen, embedding_dim)
        item_eb = torch.reshape(item_eb, (-1, self.seq_len, self.hidden_size))
         
        if self.add_pos:
            # 位置嵌入堆叠一个batch，然后与历史物品嵌入相加
            item_eb_add_pos = item_eb + self.position_embedding.repeat(item_eb.shape[0], 1, 1)
        else:
            item_eb_add_pos = item_eb

        # shape=(batch_size, maxlen, hidden_size*4)
        # linear1 (tanh(W1*H))
        item_hidden = self.linear1(item_eb_add_pos)
        # shape=(batch_size, maxlen, num_heads)
        # linear2 (W2*tanh(W1*H))
        item_att_w  = self.linear2(item_hidden)
        # shape=(batch_size, num_heads, maxlen)
        # (W2*tanh(W1*H)) transpose
        item_att_w  = torch.transpose(item_att_w, 2, 1).contiguous()

        # filter out test items in user histor  y
        atten_mask = torch.unsqueeze(mask, dim=1).repeat(1, self.num_heads, 1) # shape=(batch_size, num_heads, maxlen)
        paddings = torch.ones_like(atten_mask, dtype=torch.float) * (-2 ** 32 + 1) # softmax之后无限接近于0

        item_att_w = torch.where(torch.eq(atten_mask, 0), paddings, item_att_w)
        # softmax
        item_att_w = F.softmax(item_att_w, dim=-1) # 矩阵A，shape=(batch_size, num_heads, maxlen)

        # interest_emb即论文中的Vu
        interest_emb = torch.matmul(item_att_w, # shape=(batch_size, num_heads, maxlen)
                                item_eb # shape=(batch_size, maxlen, embedding_dim)
                                ) # shape=(batch_size, num_heads, embedding_dim)
        return interest_emb, item_att_w 
    # def calculate_optimized_sampled_loss(self, readout, pos_items, neighbor_common_user_matrix):
    #     """
    #     Args:
    #         readout: (Batch_Size, Hidden_Size)
    #         pos_items: (Batch_Size,)
    #         neighbor_common_user_matrix: SparseTensor (CPU)
    #     """
    #     batch_size = readout.shape[0]
    #     num_neg = self.sampled_n
    #     device = readout.device  # 模型所在的設備 (GPU)
        
    #     # 1. 負採樣
    #     # 注意：這裡生成的 neg_items 默認在 GPU 上
    #     neg_items = torch.randint(0, self.item_num, (batch_size, num_neg), device=device)
        
    #     # 合併索引 (GPU)
    #     all_sample_ids = torch.cat([pos_items.unsqueeze(1), neg_items], dim=1)
    #     flat_sample_ids = all_sample_ids.view(-1)

    #     # 2. Process Optimization (混合設備切片)
    #     # -------------------------------------------------------------
    #     # 檢查設備：如果矩陣在 CPU 而索引在 GPU，將索引移至 CPU
    #     matrix_device = neighbor_common_user_matrix.device()
    #     if matrix_device != flat_sample_ids.device:
    #         flat_sample_ids = flat_sample_ids.to(matrix_device)
        
    #     # 執行切片 (在 CPU 上進行)
    #     # A_subset 是一個較小的 SparseTensor
    #     A_subset = neighbor_common_user_matrix[flat_sample_ids]
        
    #     # 將切片結果移回 GPU 進行矩陣乘法
    #     if A_subset.device() != device:
    #         A_subset = A_subset.to(device)
            
    #     # 矩陣乘法 (GPU): (Batch * Sample_Num, Num_Items) @ (Num_Items, Hidden)
    #     sub_enhanced_emb = A_subset.matmul(self.embeddings.weight)
        
    #     # 3. 計算 Loss
    #     # -------------------------------------------------------------
    #     sub_enhanced_emb = sub_enhanced_emb.view(batch_size, 1 + num_neg, -1)
        
    #     logits = (readout.unsqueeze(1) * sub_enhanced_emb).sum(dim=-1)
        
    #     labels = torch.zeros(batch_size, dtype=torch.long, device=device)
    #     loss = self.loss_fct(logits, labels)
        
    #     return loss
    def calculate_optimized_sampled_loss(self, readout, pos_items, neighbor_common_user_matrix):
        """
        RTX 3090 優化版: 全 GPU 計算
        """
        batch_size = readout.shape[0]
        num_neg = self.sampled_n
        device = readout.device
        
        # 1. 負採樣 (全 GPU 操作)
        neg_items = torch.randint(0, self.item_num, (batch_size, num_neg), device=device)
        
        # 合併索引 (全 GPU 操作)
        # (Batch_Size, 1 + num_neg)
        all_sample_ids = torch.cat([pos_items.unsqueeze(1), neg_items], dim=1)
        flat_sample_ids = all_sample_ids.view(-1) # (Batch * (1+Neg))

        # 2. Process Optimization (極速版)
        # -------------------------------------------------------------
        # 因為 neighbor_common_user_matrix 已經在 GPU 上 (由 train 函數保證)
        # 這裡的切片操作會直接調用 CUDA Kernel，速度極快
        
        # [關鍵] 直接切片，不做任何 .to(cpu)
        # 假設 neighbor_common_user_matrix 是 torch_sparse.SparseTensor
        A_subset = neighbor_common_user_matrix[flat_sample_ids]
        
        # 矩陣乘法 (GPU): Sparse @ Dense -> Dense
        # 結果 shape: (Batch * Sample_Num, Hidden)
        sub_enhanced_emb = A_subset.matmul(self.embeddings.weight)
        
        # 3. 計算 Loss
        # -------------------------------------------------------------
        sub_enhanced_emb = sub_enhanced_emb.view(batch_size, 1 + num_neg, -1)
        
        # (Batch, 1, Hidden) * (Batch, 1+Neg, Hidden) -> (Batch, 1+Neg, Hidden) -> Sum last dim -> (Batch, 1+Neg)
        logits = (readout.unsqueeze(1) * sub_enhanced_emb).sum(dim=-1)
        
        # Cross Entropy Logic for Sampled Softmax
        # 正樣本在 index 0，所以 label 全是 0
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # 這裡假設 self.loss_fct 是 CrossEntropyLoss
        loss = self.loss_fct(logits, labels)
        
        return loss

