"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Uniter for VCR model
"""
import torch
from transformers.utils import logging
from torch.nn import functional as F
from torch.nn.init import xavier_normal_
logger = logging.get_logger(__name__)
from collections import defaultdict
from torch import nn
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
from .model import (
    UniterPreTrainedModel, UniterModel)
import numpy as np
class UniterForVisualCommonsenseReasoning(UniterPreTrainedModel):
    """ Finetune UNITER for VCR
    """
    def __init__(self, config, img_dim):
        super().__init__(config, img_dim)
        self.uniter = UniterModel(config, img_dim)
        self.vcr_output = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size*2),
            nn.ReLU(),
            LayerNorm(config.hidden_size*2, eps=1e-12),
            nn.Linear(config.hidden_size*2, 2)
        )
        self.apply(self.init_weights)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    def init_type_embedding(self):
        new_emb = nn.Embedding(4, self.uniter.config.hidden_size)
        new_emb.apply(self.init_weights)
        for i in [0, 1]:
            emb = self.uniter.embeddings.token_type_embeddings.weight.data[i, :]
            new_emb.weight.data[i, :].copy_(emb)
        emb = self.uniter.embeddings.token_type_embeddings.weight.data[0, :]
        new_emb.weight.data[2, :].copy_(emb)
        new_emb.weight.data[3, :].copy_(emb)
        self.uniter.embeddings.token_type_embeddings = new_emb

    def init_word_embedding(self, num_special_tokens):
        orig_word_num = self.uniter.embeddings.word_embeddings.weight.size(0)
        new_emb = nn.Embedding(
            orig_word_num + num_special_tokens, self.uniter.config.hidden_size)
        new_emb.apply(self.init_weights)
        emb = self.uniter.embeddings.word_embeddings.weight.data
        new_emb.weight.data[:orig_word_num, :].copy_(emb)
        self.uniter.embeddings.word_embeddings = new_emb

    def forward(self, batch, compute_loss=False):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attn_masks = batch['attn_masks']
        gather_index = batch['gather_index']
        txt_type_ids = batch['txt_type_ids']
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attn_masks, gather_index,
                                      output_all_encoded_layers=False,
                                      txt_type_ids=txt_type_ids)
        pooled_output = self.uniter.pooler(sequence_output)
        rank_scores = self.vcr_output(pooled_output)
        targets = batch['a_targets']
        loss =   F.cross_entropy(
                    rank_scores, targets.squeeze(-1),
                    reduction='mean')
        rank_scores=rank_scores[:,1:]
        out=rank_scores.view(rank_scores.shape[0]//4,-1)
        tar=targets.view(targets.shape[0]//4,-1)
        output=out.max(dim=-1)[1]
        ans=np.nonzero(tar)[:,1]
        matched_qa = output == ans
        return rank_scores,loss,matched_qa

class UniterForVisualCommonsenseReasoning_match(UniterPreTrainedModel):
    """ Finetune UNITER for VCR
    """
    def __init__(self, config, img_dim):
        super().__init__(config, img_dim)
        self.uniter = UniterModel(config, img_dim)
        self.hidden_size=config.hidden_size
        self.vcr_output = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size*2),
            nn.ReLU(),
            LayerNorm(config.hidden_size*2, eps=1e-12),
            nn.Linear(config.hidden_size*2, 2)
        )
        self.vcr_output_match = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.ReLU(),
            LayerNorm(config.hidden_size * 2, eps=1e-12),
            nn.Linear(config.hidden_size * 2, 4)
        )
        self.dense_avg = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.activation = nn.Tanh()
        self.apply(self.init_weights)
        # self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')


    def init_type_embedding(self):
        new_emb = nn.Embedding(4, self.uniter.config.hidden_size)
        new_emb.apply(self.init_weights)
        for i in [0, 1]:
            emb = self.uniter.embeddings.token_type_embeddings.weight.data[i, :]
            new_emb.weight.data[i, :].copy_(emb)
        emb = self.uniter.embeddings.token_type_embeddings.weight.data[0, :]
        new_emb.weight.data[2, :].copy_(emb)
        new_emb.weight.data[3, :].copy_(emb)
        self.uniter.embeddings.token_type_embeddings = new_emb

    def init_word_embedding(self, num_special_tokens):
        orig_word_num = self.uniter.embeddings.word_embeddings.weight.size(0)
        new_emb = nn.Embedding(
            orig_word_num + num_special_tokens, self.uniter.config.hidden_size)
        new_emb.apply(self.init_weights)
        emb = self.uniter.embeddings.word_embeddings.weight.data
        new_emb.weight.data[:orig_word_num, :].copy_(emb)
        self.uniter.embeddings.word_embeddings = new_emb

    def forward(self, batch, compute_loss=False):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attn_masks = batch['attn_masks']
        gather_index = batch['gather_index']
        txt_type_ids = batch['txt_type_ids']
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attn_masks, gather_index,
                                      output_all_encoded_layers=False,
                                      txt_type_ids=txt_type_ids)
        pooled_output = self.uniter.pooler(sequence_output)
        rank_scores = self.vcr_output(pooled_output)
        targets = batch['a_targets']
        cls_mask=targets!=2
        #2用来区分是不是做match
        cls_targets = torch.masked_select(targets, cls_mask)
        cls_mask_score=cls_mask.expand(cls_mask.shape[0],2)
        cls_rank_scores=torch.masked_select(rank_scores, cls_mask_score).reshape(-1,2)
        loss =   F.cross_entropy(
                    cls_rank_scores, cls_targets,
                    reduction='mean')
        cls_rank_scores=cls_rank_scores[:,1:]
        out=cls_rank_scores.view(cls_rank_scores.shape[0]//4,-1)
        tar=cls_targets.view(cls_targets.shape[0]//4,-1)
        output=out.max(dim=-1)[1]
        ans=np.nonzero(tar)[:,1]
        matched_qa = output == ans
        match_iter=batch['match_iter']
        match_iter=match_iter[::5,:]
        #重复了5遍
        ans_mask=batch['ans_mask'][4::5,:]
        match_pooler=self.match_pooler(sequence_output[4::5,:],ans_mask)
        match_scores=self.vcr_output_match(match_pooler)
        match_loss=F.cross_entropy(match_scores.reshape(match_scores.shape[0]*match_scores.shape[1],-1),match_iter.reshape(-1),reduction='mean')
        return rank_scores,loss,match_loss,matched_qa

    def match_pooler(self,sequence_output,ans_mask):
        #输入是bs*4,需要变换为坐标点
        first_token_tensor = sequence_output[:, 0]
        q_num=sequence_output.shape[0]
        pad = torch.zeros((q_num, sequence_output.shape[1] - ans_mask.shape[1]), dtype=torch.int64).cuda()
        ans_mask = torch.cat((ans_mask, pad), dim=1)
        ans_mask = ans_mask.unsqueeze(-1).expand((ans_mask.shape[0], ans_mask.shape[1], sequence_output.shape[-1]))
        ans_tensor = []
        for i in range(1, 5):
            mask = ans_mask == i
            ans_token = torch.masked_select(sequence_output, mask)
            ans_token = ans_token.view(-1, sequence_output.shape[-1])
            ans_len = mask[:, :, 0].sum(dim=1)
            cur_len = 0
            ans_mean = torch.zeros((q_num, ans_token.shape[1]), dtype=ans_token.dtype).cuda()
            for i in range(len(ans_len)):
                ans_mean[i] = ans_token[cur_len:cur_len + ans_len[i]].mean(dim=0)
                cur_len += ans_len[i]
            ans_tensor.append(ans_mean)
        ans_tensor = torch.stack(ans_tensor, dim=1)
        first_token_tensor = first_token_tensor.unsqueeze(1).expand(sequence_output.shape[0], 4, sequence_output.shape[-1])
        first_ans_token = torch.cat((first_token_tensor, ans_tensor), dim=-1)
        avg_pooled_output = self.dense_avg(first_ans_token)
        avg_pooled_output = self.activation(avg_pooled_output)
        return avg_pooled_output

class UniterForVisualCommonsenseReasoning_inf(UniterPreTrainedModel):
    """ Finetune UNITER for VCR
    """
    def __init__(self, config, img_dim):
        super().__init__(config, img_dim)
        self.uniter = UniterModel(config, img_dim)
        self.vcr_output = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size*2),
            nn.ReLU(),
            LayerNorm(config.hidden_size*2, eps=1e-12),
            nn.Linear(config.hidden_size*2, 2)
        )
        self.apply(self.init_weights)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    def init_type_embedding(self):
        new_emb = nn.Embedding(4, self.uniter.config.hidden_size)
        new_emb.apply(self.init_weights)
        for i in [0, 1]:
            emb = self.uniter.embeddings.token_type_embeddings.weight.data[i, :]
            new_emb.weight.data[i, :].copy_(emb)
        emb = self.uniter.embeddings.token_type_embeddings.weight.data[0, :]
        new_emb.weight.data[2, :].copy_(emb)
        new_emb.weight.data[3, :].copy_(emb)
        self.uniter.embeddings.token_type_embeddings = new_emb

    def init_word_embedding(self, num_special_tokens):
        orig_word_num = self.uniter.embeddings.word_embeddings.weight.size(0)
        new_emb = nn.Embedding(
            orig_word_num + num_special_tokens, self.uniter.config.hidden_size)
        new_emb.apply(self.init_weights)
        emb = self.uniter.embeddings.word_embeddings.weight.data
        new_emb.weight.data[:orig_word_num, :].copy_(emb)
        self.uniter.embeddings.word_embeddings = new_emb

    def forward(self, batch,task):
        batch = defaultdict(lambda: None, batch)
        qa_token = batch['qa_token']
        qa_num=len(batch)//5
        qar_num=len(batch)-qa_num
        assert qa_num*4==qar_num
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attn_masks = batch['attn_masks']
        gather_index = batch['gather_index']
        txt_type_ids = batch['txt_type_ids']
        if task=='qa':
            qa_mask = qa_token == 1
            input_ids=torch.masked_select(input_ids,qa_mask).reshape(qa_num,-1)
            position_ids = torch.masked_select(position_ids, qa_mask).reshape(qa_num, -1)
            img_feat = torch.masked_select(img_feat, qa_mask).reshape(qa_num, -1)
            img_pos_feat = torch.masked_select(img_pos_feat, qa_mask).reshape(qa_num, -1)
            attn_masks = torch.masked_select(attn_masks, qa_mask).reshape(qa_num, -1)
            gather_index = torch.masked_select(gather_index, qa_mask).reshape(qa_num, -1)
            txt_type_ids = torch.masked_select(txt_type_ids, qa_mask).reshape(qa_num, -1)
            sequence_output = self.uniter(input_ids, position_ids,
                                          img_feat, img_pos_feat,
                                          attn_masks, gather_index,
                                          output_all_encoded_layers=False,
                                          txt_type_ids=txt_type_ids)
            pooled_output = self.uniter.pooler(sequence_output)
            rank_scores = self.vcr_output(pooled_output)
        else:
            qar_mask = qa_token == 0
            input_ids = torch.masked_select(input_ids, qar_mask).reshape(qar_num, -1)
            position_ids = torch.masked_select(position_ids, qar_mask).reshape(qar_num, -1)
            img_feat = torch.masked_select(img_feat, qar_mask).reshape(qar_num, -1)
            img_pos_feat = torch.masked_select(img_pos_feat, qar_mask).reshape(qar_num, -1)
            attn_masks = torch.masked_select(attn_masks, qar_mask).reshape(qar_num, -1)
            gather_index = torch.masked_select(gather_index, qar_mask).reshape(qar_num, -1)
            txt_type_ids = torch.masked_select(txt_type_ids, qar_mask).reshape(qar_num, -1)
            sequence_output = self.uniter(input_ids, position_ids,
                                          img_feat, img_pos_feat,
                                          attn_masks, gather_index,
                                          output_all_encoded_layers=False,
                                          txt_type_ids=txt_type_ids)
            pooled_output = self.uniter.pooler(sequence_output)
            rank_scores = self.vcr_output(pooled_output)
        return rank_scores

class Uniter_Four_two(UniterPreTrainedModel):
    """ Finetune UNITER for VCR
    """
    def __init__(self, config, img_dim):
        super().__init__(config, img_dim)
        self.uniter = UniterModel(config, img_dim)
        self.vcr_output = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size*2),
            nn.ReLU(),
            LayerNorm(config.hidden_size*2, eps=1e-12),
            nn.Linear(config.hidden_size*2, 2)
        )
        self.apply(self.init_weights)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')

        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.activation = nn.Tanh()

    def init_type_embedding(self):
        new_emb = nn.Embedding(4, self.uniter.config.hidden_size)
        new_emb.apply(self.init_weights)
        for i in [0, 1]:
            emb = self.uniter.embeddings.token_type_embeddings.weight.data[i, :]
            new_emb.weight.data[i, :].copy_(emb)
        emb = self.uniter.embeddings.token_type_embeddings.weight.data[0, :]
        new_emb.weight.data[2, :].copy_(emb)
        new_emb.weight.data[3, :].copy_(emb)
        self.uniter.embeddings.token_type_embeddings = new_emb

    def init_word_embedding(self, num_special_tokens):
        orig_word_num = self.uniter.embeddings.word_embeddings.weight.size(0)
        new_emb = nn.Embedding(
            orig_word_num + num_special_tokens, self.uniter.config.hidden_size)
        new_emb.apply(self.init_weights)
        emb = self.uniter.embeddings.word_embeddings.weight.data
        new_emb.weight.data[:orig_word_num, :].copy_(emb)
        self.uniter.embeddings.word_embeddings = new_emb

    def forward(self, batch, compute_loss=False):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attn_masks = batch['attn_masks']
        gather_index = batch['gather_index']
        txt_type_ids = batch['txt_type_ids']
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attn_masks, gather_index,
                                      output_all_encoded_layers=False,
                                      txt_type_ids=txt_type_ids)
        ans_index=batch['ans_index']
        pooled_output = self.pooler(sequence_output,ans_index)
        rank_scores = self.vcr_output(pooled_output)
        targets = batch['a_targets']
        loss =F.cross_entropy(
                    rank_scores.reshape(rank_scores.shape[0]*rank_scores.shape[1],-1), targets.view(-1),
                    reduction='mean')
        rank_scores_soft=F.softmax(rank_scores,dim=-1)
        # rank_scores_one=rank_scores_soft[:,:,1:]
        # out=rank_scores.view(rank_scores.shape[0]//4,-1)
        # label=batch['a_label']
        # output = rank_scores_one.max(dim=1)[1].squeeze()
        rank_scores_out=rank_scores_soft.max(dim=2)[1]
        matched_qa_tmp = rank_scores_out == targets
        matched_qa=matched_qa_tmp.all(dim=1)
        return rank_scores,loss,matched_qa

    def pooler(self, hidden_states,ans_index):
        pad=torch.zeros((ans_index.shape[0],hidden_states.shape[1]-ans_index.shape[1]),dtype=torch.int64).cuda()
        ans_index=torch.cat((ans_index,pad),dim=1)
        ans_index=ans_index.unsqueeze(-1).expand((ans_index.shape[0],ans_index.shape[1],hidden_states.shape[-1]))
        mask=ans_index>0
        first_token_tensor = hidden_states[:, 0]
        ans_token=torch.masked_select(hidden_states,mask)
        ans_token=ans_token.view((hidden_states.shape[0],5,hidden_states.shape[-1]))
        ans_token=ans_token[:,1:,:]
        #取每个问题后面的SEP对应操作
        first_token_tensor=first_token_tensor.unsqueeze(1).expand(hidden_states.shape[0],4,hidden_states.shape[-1])
        first_ans_token=torch.cat((first_token_tensor,ans_token),dim=-1)
        pooled_output = self.dense(first_ans_token)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class Uniter_Four(UniterPreTrainedModel):
    """ Finetune UNITER for VCR
    """
    def __init__(self, config, img_dim):
        super().__init__(config, img_dim)
        self.uniter = UniterModel(config, img_dim)
        self.vcr_output = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size*2),
            nn.ReLU(),
            LayerNorm(config.hidden_size*2, eps=1e-12),
            nn.Linear(config.hidden_size*2, 4)
        )
        self.apply(self.init_weights)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def init_type_embedding(self):
        new_emb = nn.Embedding(4, self.uniter.config.hidden_size)
        new_emb.apply(self.init_weights)
        for i in [0, 1]:
            emb = self.uniter.embeddings.token_type_embeddings.weight.data[i, :]
            new_emb.weight.data[i, :].copy_(emb)
        emb = self.uniter.embeddings.token_type_embeddings.weight.data[0, :]
        new_emb.weight.data[2, :].copy_(emb)
        new_emb.weight.data[3, :].copy_(emb)
        self.uniter.embeddings.token_type_embeddings = new_emb

    def init_word_embedding(self, num_special_tokens):
        orig_word_num = self.uniter.embeddings.word_embeddings.weight.size(0)
        new_emb = nn.Embedding(
            orig_word_num + num_special_tokens, self.uniter.config.hidden_size)
        new_emb.apply(self.init_weights)
        emb = self.uniter.embeddings.word_embeddings.weight.data
        new_emb.weight.data[:orig_word_num, :].copy_(emb)
        self.uniter.embeddings.word_embeddings = new_emb

    def forward(self, batch, compute_loss=False):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attn_masks = batch['attn_masks']
        gather_index = batch['gather_index']
        txt_type_ids = batch['txt_type_ids']
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attn_masks, gather_index,
                                      output_all_encoded_layers=False,
                                      txt_type_ids=txt_type_ids)
        ans_index=batch['ans_index']
        pooled_output = self.pooler(sequence_output)
        rank_scores = self.vcr_output(pooled_output)
        targets = batch['a_label']
        loss =F.cross_entropy(rank_scores, targets,reduction='mean')
        output=rank_scores.max(dim=1)[1]
        matched_qa = output == targets
        return rank_scores,loss,matched_qa

    def pooler(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class Uniter_Four_match(UniterPreTrainedModel):
    """ Finetune UNITER for VCR
    """
    def __init__(self, config, img_dim):
        super().__init__(config, img_dim)
        self.uniter = UniterModel(config, img_dim)
        self.vcr_output = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size*2),
            nn.ReLU(),
            LayerNorm(config.hidden_size*2, eps=1e-12),
            nn.Linear(config.hidden_size*2, 4)
        )
        self.vcr_output_match = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size*2),
            nn.ReLU(),
            LayerNorm(config.hidden_size*2, eps=1e-12),
            nn.Linear(config.hidden_size*2, 4)
        )
        self.apply(self.init_weights)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense_avg = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.activation = nn.Tanh()

    def init_type_embedding(self):
        new_emb = nn.Embedding(4, self.uniter.config.hidden_size)
        new_emb.apply(self.init_weights)
        for i in [0, 1]:
            emb = self.uniter.embeddings.token_type_embeddings.weight.data[i, :]
            new_emb.weight.data[i, :].copy_(emb)
        emb = self.uniter.embeddings.token_type_embeddings.weight.data[0, :]
        new_emb.weight.data[2, :].copy_(emb)
        new_emb.weight.data[3, :].copy_(emb)
        self.uniter.embeddings.token_type_embeddings = new_emb

    def init_word_embedding(self, num_special_tokens):
        orig_word_num = self.uniter.embeddings.word_embeddings.weight.size(0)
        new_emb = nn.Embedding(
            orig_word_num + num_special_tokens, self.uniter.config.hidden_size)
        new_emb.apply(self.init_weights)
        emb = self.uniter.embeddings.word_embeddings.weight.data
        new_emb.weight.data[:orig_word_num, :].copy_(emb)
        self.uniter.embeddings.word_embeddings = new_emb

    def forward(self, batch, compute_loss=False):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attn_masks = batch['attn_masks']
        gather_index = batch['gather_index']
        txt_type_ids = batch['txt_type_ids']
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attn_masks, gather_index,
                                      output_all_encoded_layers=False,
                                      txt_type_ids=txt_type_ids)
        ans_mask=batch['ans_mask']
        pooled_output,avg_pooled_output = self.pooler(sequence_output,ans_mask)
        rank_scores = self.vcr_output(pooled_output)
        match_scores=self.vcr_output_match(avg_pooled_output)
        labels = batch['a_label']
        loss =F.cross_entropy(rank_scores, labels,reduction='mean')
        match_iter=batch['match_iter']
        match_loss=F.cross_entropy(match_scores.reshape(match_scores.shape[0]*match_scores.shape[1],-1),match_iter.view(-1),reduction='mean')
        output=rank_scores.max(dim=1)[1]
        matched_qa = output == labels
        return rank_scores,loss,match_loss,matched_qa

    def pooler(self, hidden_states,ans_mask):
        bs=hidden_states.shape[0]
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        pad = torch.zeros((bs, hidden_states.shape[1] - ans_mask.shape[1]), dtype=torch.int64).cuda()
        ans_mask = torch.cat((ans_mask, pad), dim=1)
        ans_mask = ans_mask.unsqueeze(-1).expand((ans_mask.shape[0], ans_mask.shape[1], hidden_states.shape[-1]))
        ans_tensor = []
        for i in range(1, 5):
            mask = ans_mask == i
            ans_token = torch.masked_select(hidden_states, mask)
            ans_token = ans_token.view(-1, hidden_states.shape[-1])
            ans_len = mask[:, :, 0].sum(dim=1)
            cur_len=0
            ans_mean=torch.zeros((bs,ans_token.shape[1]),dtype=ans_token.dtype).cuda()
            for i in range(len(ans_len)):
                ans_mean[i]=ans_token[cur_len:cur_len+ans_len[i]].mean(dim=0)
                cur_len+=ans_len[i]
            ans_tensor.append(ans_mean)
        ans_tensor=torch.stack(ans_tensor,dim=1)
        first_token_tensor = first_token_tensor.unsqueeze(1).expand(hidden_states.shape[0], 4, hidden_states.shape[-1])
        first_ans_token = torch.cat((first_token_tensor, ans_tensor), dim=-1)
        avg_pooled_output = self.dense_avg(first_ans_token)
        avg_pooled_output = self.activation(avg_pooled_output)
        return pooled_output,avg_pooled_output