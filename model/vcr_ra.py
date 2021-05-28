"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Uniter for VCR model
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from queue import PriorityQueue
# from .layer import GELU
from .model import (
    UniterPreTrainedModel, UniterModel,UniterModel_add,UniterModel_graph,GRU_encoder)
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2_INPUTS_DOCSTRING,_TOKENIZER_FOR_DOC,BaseModelOutputWithPastAndCrossAttentions,_CONFIG_FOR_DOC
from transformers.utils import logging
from transformers import GPT2LMHeadModel
from collections import defaultdict
from transformers.models.gpt2 import GPT2Model
import operator
from utils.make_dict import make_dict
from torch.nn.init import xavier_normal_
_CONFIG_FOR_DOC = "GPT2Config"
_TOKENIZER_FOR_DOC = "GPT2Tokenizer"
logger = logging.get_logger(__name__)
from collections import defaultdict
from torch import nn
from torch.nn import functional as F
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
from .model import (
    UniterPreTrainedModel, UniterModel)
class pretrain_UniterForVisualCommonsenseReasoning(UniterPreTrainedModel):
    """ Finetune UNITER for VCR
    """
    def __init__(self, config, img_dim):
        super().__init__(config, img_dim)
        self.uniter = UniterModel_graph(config, img_dim)
        self.vcr_output = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size*2),
            nn.ReLU(),
            LayerNorm(config.hidden_size*2, eps=1e-12),
            nn.Linear(config.hidden_size*2, 4)
        )
        self.apply(self.init_weights)

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

    def forward(self, batch):
        batch = defaultdict(lambda: None, batch)


        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        input_ids_golden = batch['input_ids_golden']
        attn_masks_golden = batch['attn_masks_golden']
        gather_index_golden = batch['gather_index_golden']
        txt_type_ids_golden = batch['txt_type_ids_golden']
        position_ids_golden = batch['position_ids_golden']
        vis_edge=batch['vis_edge']
        input_region_golden=batch['input_region_golden']
        sequence_output= self.uniter(input_ids_golden, position_ids_golden,
                                             img_feat, img_pos_feat,
                                             attn_masks_golden, gather_index_golden,
                                             output_all_encoded_layers=False,
                                             txt_type_ids=txt_type_ids_golden,vis_edge=vis_edge,input_region=input_region_golden)
        pooled_output = self.uniter.pooler(sequence_output)
        rank_scores= self.vcr_output(pooled_output)
        outputs=rank_scores.max(dim=1)[1]
        matched_qa = outputs == batch['a_targets']
        return rank_scores

class GPT_decoder(nn.Module):
    def __init__(self,
                 d_model=512,
                 num_layer_0=4,
                 dropout=0.2,
                 embed_dim=1024,
                 h=8,
                 d_ff=2048
                 ):
        super(GPT_decoder, self).__init__()
        self.w2i,self.i2w = make_dict()
        self.vocab_num = len(self.w2i.keys())
        self.embedding = nn.Embedding(self.vocab_num, embed_dim)
        self.dropout = dropout
        self.config_across = GPT2Config(vocab_size=29077, n_positions=512, n_ctx=512,
                                        n_layer=8,
                                        n_embd=1024, add_cross_attention=True, n_head=8)
        self.GPT_decoder = GPT2Model(self.config_across)
        self.project = nn.Linear(embed_dim, self.vocab_num,bias=False)
        self.max_len=100
        self.beam_size=3
        self.score = nn.Linear(embed_dim, 4, bias=False)

        for par in self.GPT_decoder.parameters():
            if par.ndim > 2:
                xavier_normal_(par)

    def forward(self, batch,enc_hid):
        x=batch['r_targets']
        x_lengths=batch['r_length']
        enc_attn= batch['attn_masks']
        word_embed = self.embedding(x)
        attention_mask = torch.zeros(x.size(), dtype=torch.int).cuda()
        for batch_index in range(x.size(0)):
            for sent_len in range(x_lengths[batch_index]):
                attention_mask[batch_index, sent_len] = 1
        out=self.GPT_decoder(inputs_embeds=word_embed,attention_mask=attention_mask,encoder_hidden_states=enc_hid,encoder_attention_mask=enc_attn)
        hidden = out.last_hidden_state

        logits_gen = self.project(hidden)

        logits = self.score(out.last_hidden_state)

        pooled_logits = torch.zeros((logits.size(0),logits.size(-1)),dtype=logits.dtype).cuda()
        for i in range(logits.size(0)):
            pooled_logits[i]=logits[i,x_lengths[i]-2,:]
        #取最后时间步
        return logits_gen,pooled_logits

    def predict(self,batch,enc_hid):

        trg=batch['r_targets']
        predict_his = torch.zeros((trg.size(0),self.max_len), dtype=int).cuda()
        outputs=torch.zeros((trg.size(0),self.max_len), dtype=int).cuda()
        predict_his[:, 0] =trg[:, 0]
        enc_attn = batch['attn_masks']

        for index in range(0, self.max_len-1):
            word_embed = self.embedding(predict_his[:, :index + 1])
            out = self.GPT_decoder(inputs_embeds=word_embed, encoder_hidden_states=enc_hid,encoder_attention_mask=enc_attn)
            out = out.last_hidden_state
            out = out[:, -1:, :]
            out = self.project(out)
            label = self.choose_top_word(out)
            predict_his[:,index+1]=label
            outputs[:,index]=label
        return outputs

    def choose_top_word(self, prob):
        label = np.argmax(prob.cpu().numpy(), axis=2)
        label.resize(prob.size(0))
        label = torch.from_numpy(label)
        return label

    def batch_predict_beam(self,batch,sequence_output):
        attn_masks=batch['attn_masks']
        batch_size = sequence_output.size(0)
        beam_scores = torch.zeros((batch_size, self.beam_size))  # 定义scores向量，保存累加的log_probs
        beam_scores[:, 1:] = -1e9  # 需要初始化为-inf
        beam_scores = beam_scores.view(-1)  # 展开为(batch_size * num_beams)
        done = [False for _ in range(batch_size)]  # 标记每个输入句子的beam search是否完成
        generated_hyps = [
            BeamHypotheses(self.beam_size, self.max_len, length_penalty=0.7)
            for _ in range(batch_size)
        ]  # 为每个输入句子定义维护其beam search序列的类实例
        # 初始输入: （batch_size * num_beams, 1）个sos token
        input_ids = torch.full((batch_size * self.beam_size, 1), 101, dtype=torch.long).cuda()
        cur_len=1
        sequence_output=sequence_output.unsqueeze(1)
        sequence_output_ex=sequence_output.expand(batch_size, self.beam_size, sequence_output.size(2),sequence_output.size(3))
        sequence_output_ex=sequence_output_ex.reshape(-1,sequence_output.size(2),sequence_output.size(3))
        attn_masks = attn_masks.unsqueeze(1)
        attn_masks_ex = attn_masks.expand(batch_size, self.beam_size, attn_masks.size(2))
        attn_masks_ex = attn_masks_ex.reshape(-1, sequence_output.size(2))
        while cur_len < self.max_len:
            word_embed = self.embedding(input_ids)
            out = self.GPT_decoder(inputs_embeds=word_embed, encoder_hidden_states=sequence_output_ex,
                                   encoder_attention_mask=attn_masks_ex)
            out = out.last_hidden_state
            out = out[:, -1:, :]
            out = self.project(out)
            out=out.squeeze(1)
            scores = F.log_softmax(out, dim=-1)  # log_softmax
            next_scores = scores + beam_scores[:, None].expand_as(scores).cuda()  # 累加上以前的scores
            next_scores = next_scores.view(
                batch_size, self.beam_size * self.vocab_num
            )  # 转成(batch_size, num_beams * vocab_size), 如上图所示
            # 取topk
            # next_scores: (batch_size, num_beams) next_tokens: (batch_size, num_beams)
            next_scores, next_tokens = torch.topk(next_scores, self.beam_size, dim=1, largest=True, sorted=True)

            next_batch_beam = []

            for batch_idx in range(batch_size):
                if done[batch_idx]:
                    # 当前batch的句子都解码完了，那么对应的num_beams个句子都继续pad
                    next_batch_beam.extend([(0, 0, 0)] * self.beam_size)  # pad the batch
                    continue
                next_sent_beam = []  # 保存三元组(beam_token_score, token_id, effective_beam_id)
                for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                        zip(next_tokens[batch_idx], next_scores[batch_idx])
                ):
                    beam_id = beam_token_id // self.vocab_num  # 1
                    token_id = beam_token_id % self.vocab_num  # 1
                    # 上面的公式计算beam_id只能输出0和num_beams-1, 无法输出在(batch_size, num_beams)中的真实id
                    # 如上图, batch_idx=0时，真实beam_id = 0或1; batch_idx=1时，真实beam_id如下式计算为2或3
                    # batch_idx=1时，真实beam_id如下式计算为4或5
                    effective_beam_id = batch_idx * self.beam_size + beam_id
                    # 如果遇到了eos, 则讲当前beam的句子(不含当前的eos)存入generated_hyp
                    if token_id.item() == 102:
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.beam_size
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        generated_hyps[batch_idx].add(
                            input_ids[effective_beam_id].clone(), beam_token_score.item(),
                        )
                        #还是需要继续加入，不然和encoder维度不统一
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))
                    else:
                        # 保存第beam_id个句子累加到当前的log_prob以及当前的token_id
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                    if len(next_sent_beam) == self.beam_size:
                        break
                        # 当前batch是否解码完所有句子
                    done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                        next_scores[batch_idx].max().item(), cur_len
                    )  # 注意这里取当前batch的所有log_prob的最大值
                    # 每个batch_idx, next_sent_beam中有num_beams个三元组(假设都不遇到eos)
                    # batch_idx循环后，extend后的结果为num_beams * batch_size个三元组
                next_batch_beam.extend(next_sent_beam)
                # 如果batch中每个句子的beam search都完成了，则停止
            if all(done):
                break
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])
            # 取出有效的input_ids, 因为有些beam_id不在beam_idx里面,
            # 因为有些beam id对应的句子已经解码完了
            input_ids = input_ids[beam_idx, :]  # (num_beams * batch_size, seq_len)
            # (num_beams * batch_size, seq_len) ==> (num_beams * batch_size, seq_len + 1)
            input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
            cur_len = cur_len + 1
        for batch_idx in range(batch_size):
            if done[batch_idx]:
                continue
            for beam_id in range(self.beam_size):
                # 对于每个batch_idx的每句beam，都执行加入add
                # 注意这里已经解码到max_length长度了，但是并没有遇到eos，故这里全部要尝试加入
                effective_beam_id = batch_idx * self.beam_size + beam_id
                final_score = beam_scores[effective_beam_id].item()
                final_tokens = input_ids[effective_beam_id]
                generated_hyps[batch_idx].add(final_tokens, final_score)
            # 经过上述步骤后，每个输入句子的类中保存着num_beams个最优序列
            # 下面选择若干最好的序列输出
            # 每个样本返回几个句子
        output_num_return_sequences_per_batch = 1
        output_batch_size = output_num_return_sequences_per_batch * batch_size
        # 记录每个返回句子的长度，用于后面pad
        sent_lengths = input_ids.new(output_batch_size)
        best = []

    # retrieve best hypotheses
        for i, hypotheses in enumerate(generated_hyps):
            # x: (score, hyp), x[0]: score
            sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
            for j in range(output_num_return_sequences_per_batch):
                effective_batch_idx = output_num_return_sequences_per_batch * i + j
                best_hyp = sorted_hyps.pop()[1]
                sent_lengths[effective_batch_idx] = len(best_hyp)
                best.append(best_hyp)
        if sent_lengths.min().item() != sent_lengths.max().item():
            sent_max_len = min(sent_lengths.max().item() + 1, self.max_len)
            # fill pad
            decoded = input_ids.new(output_batch_size, sent_max_len).fill_(0)

            # 填充内容
            for i, hypo in enumerate(best):
                decoded[i, : sent_lengths[i]] = hypo
                if sent_lengths[i] < self.max_len:
                    decoded[i, sent_lengths[i]] = 102
        else:
            # 否则直接堆叠起来
            decoded = torch.stack(best).type(torch.long)
            # (output_batch_size, sent_max_len) ==> (batch_size, sent_max_len)
        return decoded

class BeamSearchNode(object):
    def __init__(self,  previousNode, wordId, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward

class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty):
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty # 长度惩罚的指数系数
        self.num_beams = num_beams # beam size
        self.beams = [] # 存储最优序列及其累加的log_prob score
        self.worst_score = 1e9 # 将worst_score初始为无穷大。

    def __len__(self):
        return len(self.beams)

    def add(self, hyp, sum_logprobs):
        score = sum_logprobs / len(hyp) ** self.length_penalty # 计算惩罚后的score
        if len(self) < self.num_beams or score > self.worst_score:
                # 如果类没装满num_beams个序列
                # 或者装满以后，但是待加入序列的score值大于类中的最小值
                # 则将该序列更新进类中，并淘汰之前类中最差的序列
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                                # 如果没满的话，仅更新worst_score
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len):
                # 当解码到某一层后, 该层每个结点的分数表示从根节点到这里的log_prob之和
                # 此时取最高的log_prob, 如果此时候选序列的最高分都比类中最低分还要低的话
                # 那就没必要继续解码下去了。此时完成对该句子的解码，类中有num_beams个最优序列。
        if len(self) < self.num_beams:
            return False
        else:
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            ret = self.worst_score >= cur_score
            return ret

class UNITER_GPT(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 ):
        super(UNITER_GPT, self).__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.criterion_gen=torch.nn.CrossEntropyLoss(reduction='mean',ignore_index=0)


    def forward(self,batch):
        rank_scores,sequence_output=self.encoder(batch)
        logits_gen,pooled_logits=self.decoder(batch,sequence_output)
        mse_loss=self.calc_kl_div(rank_scores,pooled_logits)

        a_target=batch['a_targets']
        r_labels=batch['r_labels']
        cep_loss=self.criterion(rank_scores,a_target)
        ceg_loss=self.criterion(pooled_logits,a_target)
        mle_loss=self.criterion_gen(logits_gen.reshape(logits_gen.size(0)*logits_gen.size(1),-1),r_labels.view(-1))
        return mse_loss,cep_loss,ceg_loss,mle_loss

    def predict(self,batch):
        rank_scores,sequence_output=self.encoder(batch)
        outputs=self.decoder.predict(batch,sequence_output)
        outputs_a = rank_scores.max(dim=-1)[1]
        return outputs,outputs_a

    def batch_predict_beam(self,batch):
        rank_scores,sequence_output=self.encoder(batch)
        outputs=self.decoder.batch_predict_beam(batch,sequence_output)
        return outputs

    def calc_kl_div(self,mrc_outputs, explanation_outputs, temperature=1.0):
        loss_kl = F.kl_div(
            input=F.log_softmax(mrc_outputs / temperature, dim=-1),
            target=F.softmax(explanation_outputs / temperature, dim=-1),
            reduction="batchmean",
        ) * (temperature ** 2)

        return loss_kl

class Graph_con_UniterForVisualCommonsenseReasoning(UniterPreTrainedModel):
    """ Finetune UNITER for VCR
    """
    def __init__(self, config, img_dim):
        super().__init__(config, img_dim)
        self.uniter = UniterModel_graph(config, img_dim)
        self.vcr_output = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size*2),
            nn.ReLU(),
            LayerNorm(config.hidden_size*2, eps=1e-12),
            nn.Linear(config.hidden_size*2, 4)
        )
        self.apply(self.init_weights)

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
        input_region=batch['input_region']
        vis_edge=batch['vis_edge']
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attn_masks, gather_index,
                                      output_all_encoded_layers=False,
                                      txt_type_ids=txt_type_ids,
                                      input_region=input_region,
                                      vis_edge=vis_edge)
        # pooled_output = self.uniter.pooler(sequence_output)
        #
        # rank_scores = self.vcr_output(pooled_output)

        return sequence_output

class Graph_con_GPT_decoder(nn.Module):
    def __init__(self,
                 d_model=512,
                 num_layer_0=4,
                 dropout=0.2,
                 embed_dim=1024,
                 h=8,
                 d_ff=2048
                 ):
        super(Graph_con_GPT_decoder, self).__init__()
        self.w2i,self.i2w = make_dict()
        self.vocab_num = len(self.w2i.keys())
        self.embedding = nn.Embedding(self.vocab_num, embed_dim)
        self.dropout = dropout
        self.config_across = GPT2Config(vocab_size=29077, n_positions=512, n_ctx=512,
                                        n_layer=8,
                                        n_embd=1024, add_cross_attention=True, n_head=8)
        self.GPT_decoder = GPT2Model(self.config_across)
        self.project = nn.Linear(embed_dim, self.vocab_num,bias=False)
        self.max_len=100
        self.beam_size=3
        self.score = nn.Linear(embed_dim, 4, bias=False)

        for par in self.GPT_decoder.parameters():
            if par.ndim > 2:
                xavier_normal_(par)

    def forward(self, batch,enc_hid):
        x=batch['r_targets']
        x_lengths=batch['r_length']
        enc_attn= batch['attn_masks']
        word_embed = self.embedding(x)
        attention_mask = torch.zeros(x.size(), dtype=torch.int).cuda()
        for batch_index in range(x.size(0)):
            for sent_len in range(x_lengths[batch_index]):
                attention_mask[batch_index, sent_len] = 1
        out=self.GPT_decoder(inputs_embeds=word_embed,attention_mask=attention_mask,encoder_hidden_states=enc_hid,encoder_attention_mask=enc_attn)
        hidden = out.last_hidden_state

        logits_gen = self.project(hidden)

        logits = self.score(out.last_hidden_state)


        pooled_hidden=torch.zeros((hidden.size(0),hidden.size(-1)),dtype=hidden.dtype).cuda()
        pooled_logits = torch.zeros((logits.size(0),logits.size(-1)),dtype=logits.dtype).cuda()
        for i in range(logits.size(0)):
            pooled_logits[i]=logits[i,x_lengths[i]-2,:]
            pooled_hidden[i]=hidden[i,x_lengths[i]-2,:]
        #取最后时间步
        return logits_gen,pooled_logits,pooled_hidden

    def predict(self,batch,enc_hid):

        trg=batch['r_targets']
        predict_his = torch.zeros((trg.size(0),self.max_len), dtype=int).cuda()
        outputs=torch.zeros((trg.size(0),self.max_len), dtype=int).cuda()
        predict_his[:, 0] =trg[:, 0]
        enc_attn = batch['attn_masks']

        for index in range(0, self.max_len-1):
            word_embed = self.embedding(predict_his[:, :index + 1])
            out = self.GPT_decoder(inputs_embeds=word_embed, encoder_hidden_states=enc_hid,encoder_attention_mask=enc_attn)
            out = out.last_hidden_state
            out_last = out[:, -1:, :]
            logits = self.project(out_last)
            label = self.choose_top_word(logits)
            predict_his[:,index+1]=label
            outputs[:,index]=label
        pooled_hidden = torch.zeros((trg.size(0), 1024), dtype=torch.float32).cuda()
        out_np = outputs.cpu().numpy()
        for i in range(pooled_hidden.size(0)):
            try:
                #如果有输出102
                index=np.argwhere(out_np[i] == 102)[0]
                pooled_hidden[i] = out[i, index, :]
            except:
                #如果一直没输出102
                pooled_hidden[i] = out[i, -1, :]

        return outputs,pooled_hidden

    def choose_top_word(self, prob):
        label = np.argmax(prob.cpu().numpy(), axis=2)
        label.resize(prob.size(0))
        label = torch.from_numpy(label)
        return label

    def batch_predict_beam(self,batch,sequence_output):
        attn_masks=batch['attn_masks']
        batch_size = sequence_output.size(0)
        beam_scores = torch.zeros((batch_size, self.beam_size))  # 定义scores向量，保存累加的log_probs
        beam_scores[:, 1:] = -1e9  # 需要初始化为-inf
        beam_scores = beam_scores.view(-1)  # 展开为(batch_size * num_beams)
        done = [False for _ in range(batch_size)]  # 标记每个输入句子的beam search是否完成
        generated_hyps = [
            BeamHypotheses(self.beam_size, self.max_len, length_penalty=0.7)
            for _ in range(batch_size)
        ]  # 为每个输入句子定义维护其beam search序列的类实例
        # 初始输入: （batch_size * num_beams, 1）个sos token
        input_ids = torch.full((batch_size * self.beam_size, 1), 101, dtype=torch.long).cuda()
        cur_len=1
        sequence_output=sequence_output.unsqueeze(1)
        sequence_output_ex=sequence_output.expand(batch_size, self.beam_size, sequence_output.size(2),sequence_output.size(3))
        sequence_output_ex=sequence_output_ex.reshape(-1,sequence_output.size(2),sequence_output.size(3))
        attn_masks = attn_masks.unsqueeze(1)
        attn_masks_ex = attn_masks.expand(batch_size, self.beam_size, attn_masks.size(2))
        attn_masks_ex = attn_masks_ex.reshape(-1, sequence_output.size(2))
        while cur_len < self.max_len:
            word_embed = self.embedding(input_ids)
            out = self.GPT_decoder(inputs_embeds=word_embed, encoder_hidden_states=sequence_output_ex,
                                   encoder_attention_mask=attn_masks_ex)
            out = out.last_hidden_state
            out = out[:, -1:, :]
            out = self.project(out)
            out=out.squeeze(1)
            scores = F.log_softmax(out, dim=-1)  # log_softmax
            next_scores = scores + beam_scores[:, None].expand_as(scores).cuda()  # 累加上以前的scores
            next_scores = next_scores.view(
                batch_size, self.beam_size * self.vocab_num
            )  # 转成(batch_size, num_beams * vocab_size), 如上图所示
            # 取topk
            # next_scores: (batch_size, num_beams) next_tokens: (batch_size, num_beams)
            next_scores, next_tokens = torch.topk(next_scores, self.beam_size, dim=1, largest=True, sorted=True)

            next_batch_beam = []

            for batch_idx in range(batch_size):
                if done[batch_idx]:
                    # 当前batch的句子都解码完了，那么对应的num_beams个句子都继续pad
                    next_batch_beam.extend([(0, 0, 0)] * self.beam_size)  # pad the batch
                    continue
                next_sent_beam = []  # 保存三元组(beam_token_score, token_id, effective_beam_id)
                for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                        zip(next_tokens[batch_idx], next_scores[batch_idx])
                ):
                    beam_id = beam_token_id // self.vocab_num  # 1
                    token_id = beam_token_id % self.vocab_num  # 1
                    # 上面的公式计算beam_id只能输出0和num_beams-1, 无法输出在(batch_size, num_beams)中的真实id
                    # 如上图, batch_idx=0时，真实beam_id = 0或1; batch_idx=1时，真实beam_id如下式计算为2或3
                    # batch_idx=1时，真实beam_id如下式计算为4或5
                    effective_beam_id = batch_idx * self.beam_size + beam_id
                    # 如果遇到了eos, 则讲当前beam的句子(不含当前的eos)存入generated_hyp
                    if token_id.item() == 102:
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.beam_size
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        generated_hyps[batch_idx].add(
                            input_ids[effective_beam_id].clone(), beam_token_score.item(),
                        )
                        #还是需要继续加入，不然和encoder维度不统一
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))
                    else:
                        # 保存第beam_id个句子累加到当前的log_prob以及当前的token_id
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                    if len(next_sent_beam) == self.beam_size:
                        break
                        # 当前batch是否解码完所有句子
                    done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                        next_scores[batch_idx].max().item(), cur_len
                    )  # 注意这里取当前batch的所有log_prob的最大值
                    # 每个batch_idx, next_sent_beam中有num_beams个三元组(假设都不遇到eos)
                    # batch_idx循环后，extend后的结果为num_beams * batch_size个三元组
                next_batch_beam.extend(next_sent_beam)
                # 如果batch中每个句子的beam search都完成了，则停止
            if all(done):
                break
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])
            # 取出有效的input_ids, 因为有些beam_id不在beam_idx里面,
            # 因为有些beam id对应的句子已经解码完了
            input_ids = input_ids[beam_idx, :]  # (num_beams * batch_size, seq_len)
            # (num_beams * batch_size, seq_len) ==> (num_beams * batch_size, seq_len + 1)
            input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
            cur_len = cur_len + 1
        for batch_idx in range(batch_size):
            if done[batch_idx]:
                continue
            for beam_id in range(self.beam_size):
                # 对于每个batch_idx的每句beam，都执行加入add
                # 注意这里已经解码到max_length长度了，但是并没有遇到eos，故这里全部要尝试加入
                effective_beam_id = batch_idx * self.beam_size + beam_id
                final_score = beam_scores[effective_beam_id].item()
                final_tokens = input_ids[effective_beam_id]
                generated_hyps[batch_idx].add(final_tokens, final_score)
            # 经过上述步骤后，每个输入句子的类中保存着num_beams个最优序列
            # 下面选择若干最好的序列输出
            # 每个样本返回几个句子
        output_num_return_sequences_per_batch = 1
        output_batch_size = output_num_return_sequences_per_batch * batch_size
        # 记录每个返回句子的长度，用于后面pad
        sent_lengths = input_ids.new(output_batch_size)
        best = []

    # retrieve best hypotheses
        for i, hypotheses in enumerate(generated_hyps):
            # x: (score, hyp), x[0]: score
            sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
            for j in range(output_num_return_sequences_per_batch):
                effective_batch_idx = output_num_return_sequences_per_batch * i + j
                best_hyp = sorted_hyps.pop()[1]
                sent_lengths[effective_batch_idx] = len(best_hyp)
                best.append(best_hyp)
        if sent_lengths.min().item() != sent_lengths.max().item():
            sent_max_len = min(sent_lengths.max().item() + 1, self.max_len)
            # fill pad
            decoded = input_ids.new(output_batch_size, sent_max_len).fill_(0)

            # 填充内容
            for i, hypo in enumerate(best):
                decoded[i, : sent_lengths[i]] = hypo
                if sent_lengths[i] < self.max_len:
                    decoded[i, sent_lengths[i]] = 102
        else:
            # 否则直接堆叠起来
            decoded = torch.stack(best).type(torch.long)
            # (output_batch_size, sent_max_len) ==> (batch_size, sent_max_len)
        return decoded

class Graph_con_UNITER_GPT(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 ):
        super(Graph_con_UNITER_GPT, self).__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.criterion_gen=torch.nn.CrossEntropyLoss(reduction='mean',ignore_index=0)
        self.a_output = nn.Sequential(
            nn.Linear(encoder.config.hidden_size, encoder.config.hidden_size * 2),
            nn.ReLU(),
            LayerNorm(encoder.config.hidden_size * 2, eps=1e-12),
            nn.Linear(encoder.config.hidden_size * 2, 4)
        )
        self.linear=nn.Linear(encoder.config.hidden_size * 2,encoder.config.hidden_size,bias=False)
        self.sig = nn.Sigmoid()

    def forward(self,batch):
        sequence_output=self.encoder(batch)
        logits_gen,pooled_logits,pooled_hidden=self.decoder(batch,sequence_output)

        encoder_pooled_hidden=torch.zeros((sequence_output.size(0),sequence_output.size(-1)),dtype=sequence_output.dtype).cuda()
        attn_mask=batch['attn_masks']
        for i in range(sequence_output.size(0)):
            len=sum(attn_mask[i])
            tmp=torch.mean(sequence_output[i,:len,:],dim=0)
            encoder_pooled_hidden[i]=tmp

        alpha=torch.cat((pooled_hidden,encoder_pooled_hidden),dim=-1)
        alpha=self.linear(alpha)
        alpha=self.sig(alpha)
        cross_hidden=alpha.mul(pooled_hidden)+(1-alpha).mul(encoder_pooled_hidden)
        rank_scores=self.a_output(cross_hidden)

        mse_loss=self.calc_kl_div(rank_scores,pooled_logits)

        a_target=batch['a_targets']
        r_labels=batch['r_labels']
        cep_loss=self.criterion(rank_scores,a_target)
        ceg_loss=self.criterion(pooled_logits,a_target)
        mle_loss=self.criterion_gen(logits_gen.reshape(logits_gen.size(0)*logits_gen.size(1),-1),r_labels.view(-1))
        return mse_loss,cep_loss,ceg_loss,mle_loss

    def predict(self,batch):
        sequence_output=self.encoder(batch)
        outputs,pooled_hidden=self.decoder.predict(batch,sequence_output)

        encoder_pooled_hidden = torch.zeros((sequence_output.size(0), sequence_output.size(-1)),
                                            dtype=sequence_output.dtype).cuda()
        attn_mask = batch['attn_masks']
        for i in range(sequence_output.size(0)):
            len = sum(attn_mask[i])
            tmp = torch.mean(sequence_output[i, :len, :], dim=0)
            encoder_pooled_hidden[i] = tmp

        alpha = torch.cat((pooled_hidden, encoder_pooled_hidden), dim=-1)
        alpha = self.linear(alpha)
        alpha = self.sig(alpha)
        cross_hidden = alpha.mul(pooled_hidden) + (1 - alpha).mul(encoder_pooled_hidden)
        rank_scores = self.a_output(cross_hidden)
        outputs_a = rank_scores.max(dim=-1)[1]
        return outputs,outputs_a

    def batch_predict_beam(self,batch):
        sequence_output=self.encoder(batch)
        outputs=self.decoder.batch_predict_beam(batch,sequence_output)

        encoder_pooled_hidden = torch.zeros((sequence_output.size(0), sequence_output.size(-1)),
                                            dtype=sequence_output.dtype).cuda()
        attn_mask = batch['attn_masks']
        for i in range(sequence_output.size(0)):
            len = sum(attn_mask[i])
            tmp = torch.mean(sequence_output[i, :len, :], dim=0)
            encoder_pooled_hidden[i] = tmp
        return outputs

    def calc_kl_div(self,mrc_outputs, explanation_outputs, temperature=1.0):
        loss_kl = F.kl_div(
            input=F.log_softmax(mrc_outputs / temperature, dim=-1),
            target=F.softmax(explanation_outputs / temperature, dim=-1),
            reduction="batchmean",
        ) * (temperature ** 2)

        return loss_kl

class Graph_UniterForVisualCommonsenseReasoning(UniterPreTrainedModel):
    """ Finetune UNITER for VCR
    """
    def __init__(self, config, img_dim):
        super().__init__(config, img_dim)
        self.uniter = UniterModel_graph(config, img_dim)
        self.vcr_output = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size*2),
            nn.ReLU(),
            LayerNorm(config.hidden_size*2, eps=1e-12),
            nn.Linear(config.hidden_size*2, 4)
        )
        self.apply(self.init_weights)

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
        input_region=batch['input_region']
        vis_edge=batch['vis_edge']
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attn_masks, gather_index,
                                      output_all_encoded_layers=False,
                                      txt_type_ids=txt_type_ids,
                                      input_region=input_region,
                                      vis_edge=vis_edge)
        pooled_output = self.uniter.pooler(sequence_output)

        rank_scores = self.vcr_output(pooled_output)

        return sequence_output,rank_scores

class Graph_GPT_decoder(nn.Module):
    def __init__(self,
                 d_model=512,
                 num_layer_0=4,
                 dropout=0.2,
                 embed_dim=1024,
                 h=8,
                 d_ff=2048
                 ):
        super(Graph_GPT_decoder, self).__init__()
        self.w2i,self.i2w = make_dict()
        self.vocab_num = len(self.w2i.keys())
        self.embedding = nn.Embedding(self.vocab_num, embed_dim)
        self.dropout = dropout
        self.config_across = GPT2Config(vocab_size=29077, n_positions=512, n_ctx=512,
                                        n_layer=8,
                                        n_embd=1024, add_cross_attention=True, n_head=8)
        self.GPT_decoder = GPT2Model(self.config_across)
        self.project = nn.Linear(embed_dim, self.vocab_num,bias=False)
        self.max_len=100
        self.beam_size=3
        self.score = nn.Linear(embed_dim, 4, bias=False)

        for par in self.GPT_decoder.parameters():
            if par.ndim > 2:
                xavier_normal_(par)

    def forward(self, batch,enc_hid):
        x=batch['r_targets']
        x_lengths=batch['r_length']
        enc_attn= batch['attn_masks']
        word_embed = self.embedding(x)
        attention_mask = torch.zeros(x.size(), dtype=torch.int).cuda()
        for batch_index in range(x.size(0)):
            for sent_len in range(x_lengths[batch_index]):
                attention_mask[batch_index, sent_len] = 1
        out=self.GPT_decoder(inputs_embeds=word_embed,attention_mask=attention_mask,encoder_hidden_states=enc_hid,encoder_attention_mask=enc_attn)
        hidden = out.last_hidden_state

        logits_gen = self.project(hidden)

        logits = self.score(out.last_hidden_state)


        pooled_hidden=torch.zeros((hidden.size(0),hidden.size(-1)),dtype=hidden.dtype).cuda()
        pooled_logits = torch.zeros((logits.size(0),logits.size(-1)),dtype=logits.dtype).cuda()
        for i in range(logits.size(0)):
            pooled_logits[i]=logits[i,x_lengths[i]-2,:]
            pooled_hidden[i]=hidden[i,x_lengths[i]-2,:]
        #取最后时间步
        return logits_gen,pooled_logits,pooled_hidden

    def predict(self,batch,enc_hid):

        trg=batch['r_targets']
        predict_his = torch.zeros((trg.size(0),self.max_len), dtype=int).cuda()
        outputs=torch.zeros((trg.size(0),self.max_len), dtype=int).cuda()
        predict_his[:, 0] =trg[:, 0]
        enc_attn = batch['attn_masks']

        for index in range(0, self.max_len-1):
            word_embed = self.embedding(predict_his[:, :index + 1])
            out = self.GPT_decoder(inputs_embeds=word_embed, encoder_hidden_states=enc_hid,encoder_attention_mask=enc_attn)
            out = out.last_hidden_state
            out_last = out[:, -1:, :]
            logits = self.project(out_last)
            label = self.choose_top_word(logits)
            predict_his[:,index+1]=label
            outputs[:,index]=label
        pooled_hidden = torch.zeros((trg.size(0), 1024), dtype=torch.float32).cuda()
        out_np = outputs.cpu().numpy()
        for i in range(pooled_hidden.size(0)):
            try:
                #如果有输出102
                index=np.argwhere(out_np[i] == 102)[0]
                pooled_hidden[i] = out[i, index, :]
            except:
                #如果一直没输出102
                pooled_hidden[i] = out[i, -1, :]

        return outputs,pooled_hidden

    def choose_top_word(self, prob):
        label = np.argmax(prob.cpu().numpy(), axis=2)
        label.resize(prob.size(0))
        label = torch.from_numpy(label)
        return label

    def batch_predict_beam(self,batch,sequence_output):
        attn_masks=batch['attn_masks']
        batch_size = sequence_output.size(0)
        beam_scores = torch.zeros((batch_size, self.beam_size))  # 定义scores向量，保存累加的log_probs
        beam_scores[:, 1:] = -1e9  # 需要初始化为-inf
        beam_scores = beam_scores.view(-1)  # 展开为(batch_size * num_beams)
        done = [False for _ in range(batch_size)]  # 标记每个输入句子的beam search是否完成
        generated_hyps = [
            BeamHypotheses(self.beam_size, self.max_len, length_penalty=0.7)
            for _ in range(batch_size)
        ]  # 为每个输入句子定义维护其beam search序列的类实例
        # 初始输入: （batch_size * num_beams, 1）个sos token
        input_ids = torch.full((batch_size * self.beam_size, 1), 101, dtype=torch.long).cuda()
        cur_len=1
        sequence_output=sequence_output.unsqueeze(1)
        sequence_output_ex=sequence_output.expand(batch_size, self.beam_size, sequence_output.size(2),sequence_output.size(3))
        sequence_output_ex=sequence_output_ex.reshape(-1,sequence_output.size(2),sequence_output.size(3))
        attn_masks = attn_masks.unsqueeze(1)
        attn_masks_ex = attn_masks.expand(batch_size, self.beam_size, attn_masks.size(2))
        attn_masks_ex = attn_masks_ex.reshape(-1, sequence_output.size(2))
        while cur_len < self.max_len:
            word_embed = self.embedding(input_ids)
            out = self.GPT_decoder(inputs_embeds=word_embed, encoder_hidden_states=sequence_output_ex,
                                   encoder_attention_mask=attn_masks_ex)
            out = out.last_hidden_state
            out = out[:, -1:, :]
            out = self.project(out)
            out=out.squeeze(1)
            scores = F.log_softmax(out, dim=-1)  # log_softmax
            next_scores = scores + beam_scores[:, None].expand_as(scores).cuda()  # 累加上以前的scores
            next_scores = next_scores.view(
                batch_size, self.beam_size * self.vocab_num
            )  # 转成(batch_size, num_beams * vocab_size), 如上图所示
            # 取topk
            # next_scores: (batch_size, num_beams) next_tokens: (batch_size, num_beams)
            next_scores, next_tokens = torch.topk(next_scores, self.beam_size, dim=1, largest=True, sorted=True)

            next_batch_beam = []

            for batch_idx in range(batch_size):
                if done[batch_idx]:
                    # 当前batch的句子都解码完了，那么对应的num_beams个句子都继续pad
                    next_batch_beam.extend([(0, 0, 0)] * self.beam_size)  # pad the batch
                    continue
                next_sent_beam = []  # 保存三元组(beam_token_score, token_id, effective_beam_id)
                for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                        zip(next_tokens[batch_idx], next_scores[batch_idx])
                ):
                    beam_id = beam_token_id // self.vocab_num  # 1
                    token_id = beam_token_id % self.vocab_num  # 1
                    # 上面的公式计算beam_id只能输出0和num_beams-1, 无法输出在(batch_size, num_beams)中的真实id
                    # 如上图, batch_idx=0时，真实beam_id = 0或1; batch_idx=1时，真实beam_id如下式计算为2或3
                    # batch_idx=1时，真实beam_id如下式计算为4或5
                    effective_beam_id = batch_idx * self.beam_size + beam_id
                    # 如果遇到了eos, 则讲当前beam的句子(不含当前的eos)存入generated_hyp
                    if token_id.item() == 102:
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.beam_size
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        generated_hyps[batch_idx].add(
                            input_ids[effective_beam_id].clone(), beam_token_score.item(),
                        )
                        #还是需要继续加入，不然和encoder维度不统一
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))
                    else:
                        # 保存第beam_id个句子累加到当前的log_prob以及当前的token_id
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                    if len(next_sent_beam) == self.beam_size:
                        break
                        # 当前batch是否解码完所有句子
                    done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                        next_scores[batch_idx].max().item(), cur_len
                    )  # 注意这里取当前batch的所有log_prob的最大值
                    # 每个batch_idx, next_sent_beam中有num_beams个三元组(假设都不遇到eos)
                    # batch_idx循环后，extend后的结果为num_beams * batch_size个三元组
                next_batch_beam.extend(next_sent_beam)
                # 如果batch中每个句子的beam search都完成了，则停止
            if all(done):
                break
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])
            # 取出有效的input_ids, 因为有些beam_id不在beam_idx里面,
            # 因为有些beam id对应的句子已经解码完了
            input_ids = input_ids[beam_idx, :]  # (num_beams * batch_size, seq_len)
            # (num_beams * batch_size, seq_len) ==> (num_beams * batch_size, seq_len + 1)
            input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
            cur_len = cur_len + 1
        for batch_idx in range(batch_size):
            if done[batch_idx]:
                continue
            for beam_id in range(self.beam_size):
                # 对于每个batch_idx的每句beam，都执行加入add
                # 注意这里已经解码到max_length长度了，但是并没有遇到eos，故这里全部要尝试加入
                effective_beam_id = batch_idx * self.beam_size + beam_id
                final_score = beam_scores[effective_beam_id].item()
                final_tokens = input_ids[effective_beam_id]
                generated_hyps[batch_idx].add(final_tokens, final_score)
            # 经过上述步骤后，每个输入句子的类中保存着num_beams个最优序列
            # 下面选择若干最好的序列输出
            # 每个样本返回几个句子
        output_num_return_sequences_per_batch = 1
        output_batch_size = output_num_return_sequences_per_batch * batch_size
        # 记录每个返回句子的长度，用于后面pad
        sent_lengths = input_ids.new(output_batch_size)
        best = []

    # retrieve best hypotheses
        for i, hypotheses in enumerate(generated_hyps):
            # x: (score, hyp), x[0]: score
            sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
            for j in range(output_num_return_sequences_per_batch):
                effective_batch_idx = output_num_return_sequences_per_batch * i + j
                best_hyp = sorted_hyps.pop()[1]
                sent_lengths[effective_batch_idx] = len(best_hyp)
                best.append(best_hyp)
        if sent_lengths.min().item() != sent_lengths.max().item():
            sent_max_len = min(sent_lengths.max().item() + 1, self.max_len)
            # fill pad
            decoded = input_ids.new(output_batch_size, sent_max_len).fill_(0)

            # 填充内容
            for i, hypo in enumerate(best):
                decoded[i, : sent_lengths[i]] = hypo
                if sent_lengths[i] < self.max_len:
                    decoded[i, sent_lengths[i]] = 102
        else:
            # 否则直接堆叠起来
            decoded = torch.stack(best).type(torch.long)
            # (output_batch_size, sent_max_len) ==> (batch_size, sent_max_len)
        return decoded

class Graph_UNITER_GPT(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 ):
        super(Graph_UNITER_GPT, self).__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.criterion_gen=torch.nn.CrossEntropyLoss(reduction='mean',ignore_index=0)
        self.a_output = nn.Sequential(
            nn.Linear(encoder.config.hidden_size, encoder.config.hidden_size * 2),
            nn.ReLU(),
            LayerNorm(encoder.config.hidden_size * 2, eps=1e-12),
            nn.Linear(encoder.config.hidden_size * 2, 4)
        )
        self.linear=nn.Linear(encoder.config.hidden_size * 2,encoder.config.hidden_size,bias=False)
        self.sig = nn.Sigmoid()

    def forward(self,batch):
        sequence_output,rank_scores=self.encoder(batch)
        logits_gen,pooled_logits,pooled_hidden=self.decoder(batch,sequence_output)

        # encoder_pooled_hidden=torch.zeros((sequence_output.size(0),sequence_output.size(-1)),dtype=sequence_output.dtype).cuda()
        # attn_mask=batch['attn_masks']
        # for i in range(sequence_output.size(0)):
        #     len=sum(attn_mask[i])
        #     tmp=torch.mean(sequence_output[i,:len,:],dim=0)
        #     encoder_pooled_hidden[i]=tmp
        #
        # alpha=torch.cat((pooled_hidden,encoder_pooled_hidden),dim=-1)
        # alpha=self.linear(alpha)
        # alpha=self.sig(alpha)
        # cross_hidden=alpha.mul(pooled_hidden)+(1-alpha).mul(encoder_pooled_hidden)
        # rank_scores=self.a_output(cross_hidden)

        mse_loss=self.calc_kl_div(rank_scores,pooled_logits)

        a_target=batch['a_targets']
        r_labels=batch['r_labels']
        cep_loss=self.criterion(rank_scores,a_target)
        ceg_loss=self.criterion(pooled_logits,a_target)
        mle_loss=self.criterion_gen(logits_gen.reshape(logits_gen.size(0)*logits_gen.size(1),-1),r_labels.view(-1))
        return mse_loss,cep_loss,ceg_loss,mle_loss

    def predict(self,batch):
        sequence_output,rank_scores=self.encoder(batch)
        outputs,pooled_hidden=self.decoder.predict(batch,sequence_output)

        # encoder_pooled_hidden = torch.zeros((sequence_output.size(0), sequence_output.size(-1)),
        #                                     dtype=sequence_output.dtype).cuda()
        # attn_mask = batch['attn_masks']
        # for i in range(sequence_output.size(0)):
        #     len = sum(attn_mask[i])
        #     tmp = torch.mean(sequence_output[i, :len, :], dim=0)
        #     encoder_pooled_hidden[i] = tmp
        #
        # alpha = torch.cat((pooled_hidden, encoder_pooled_hidden), dim=-1)
        # alpha = self.linear(alpha)
        # alpha = self.sig(alpha)
        # cross_hidden = alpha.mul(pooled_hidden) + (1 - alpha).mul(encoder_pooled_hidden)
        # rank_scores = self.a_output(cross_hidden)
        outputs_a = rank_scores.max(dim=-1)[1]
        return outputs,outputs_a

    def batch_predict_beam(self,batch):
        sequence_output=self.encoder(batch)
        outputs=self.decoder.batch_predict_beam(batch,sequence_output)

        encoder_pooled_hidden = torch.zeros((sequence_output.size(0), sequence_output.size(-1)),
                                            dtype=sequence_output.dtype).cuda()
        attn_mask = batch['attn_masks']
        for i in range(sequence_output.size(0)):
            len = sum(attn_mask[i])
            tmp = torch.mean(sequence_output[i, :len, :], dim=0)
            encoder_pooled_hidden[i] = tmp
        return outputs

    def calc_kl_div(self,mrc_outputs, explanation_outputs, temperature=1.0):
        loss_kl = F.kl_div(
            input=F.log_softmax(mrc_outputs / temperature, dim=-1),
            target=F.softmax(explanation_outputs / temperature, dim=-1),
            reduction="batchmean",
        ) * (temperature ** 2)

        return loss_kl

class Graph_encoder_UniterForVisualCommonsenseReasoning(UniterPreTrainedModel):
    """ Finetune UNITER for VCR
    """
    def __init__(self, config, img_dim):
        super().__init__(config, img_dim)
        self.uniter = UniterModel_graph(config, img_dim)
        self.vcr_output = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size*2),
            nn.ReLU(),
            LayerNorm(config.hidden_size*2, eps=1e-12),
            nn.Linear(config.hidden_size*2, 4)
        )
        self.apply(self.init_weights)

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
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        vis_edge=batch['vis_edge']
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        attn_masks = batch['attn_masks']
        gather_index = batch['gather_index']
        txt_type_ids = batch['txt_type_ids']
        input_region=batch['input_region']
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attn_masks, gather_index,
                                      output_all_encoded_layers=False,
                                      txt_type_ids=txt_type_ids,
                                      input_region=input_region,
                                      vis_edge=vis_edge)
        pooled_output = self.uniter.pooler(sequence_output)

        rank_scores = self.vcr_output(pooled_output)

        return sequence_output,rank_scores

class Graph_golden_UniterForVisualCommonsenseReasoning(UniterPreTrainedModel):
    """ Finetune UNITER for VCR
    """
    def __init__(self, config, img_dim):
        super().__init__(config, img_dim)
        self.uniter = UniterModel_graph(config, img_dim)
        self.vcr_output = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size*2),
            nn.ReLU(),
            LayerNorm(config.hidden_size*2, eps=1e-12),
            nn.Linear(config.hidden_size*2, 4)
        )
        self.apply(self.init_weights)

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
        input_ids_golden = batch['input_ids_golden']
        position_ids_golden= batch['position_ids_golden']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attn_masks_golden = batch['attn_masks_golden']
        gather_index_golden = batch['gather_index_golden']
        txt_type_ids_golden = batch['txt_type_ids_golden']
        input_region_golden=batch['input_region_golden']
        vis_edge=batch['vis_edge']
        sequence_output_golden = self.uniter(input_ids_golden, position_ids_golden,
                                      img_feat, img_pos_feat,
                                      attn_masks_golden, gather_index_golden,
                                      output_all_encoded_layers=False,
                                      txt_type_ids=txt_type_ids_golden,
                                      input_region=input_region_golden,
                                      vis_edge=vis_edge)
        pooled_output_golden = self.uniter.pooler(sequence_output_golden)
        rank_scores_golden = self.vcr_output(pooled_output_golden)
        return rank_scores_golden

class Graph_golden_UNITER_GPT(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 golden,
                 wo_pred_loss
                 ):
        super(Graph_golden_UNITER_GPT, self).__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.golden=golden
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.criterion_gen=torch.nn.CrossEntropyLoss(reduction='mean',ignore_index=0)
        self.a_output = nn.Sequential(
            nn.Linear(encoder.config.hidden_size, encoder.config.hidden_size * 2),
            nn.ReLU(),
            LayerNorm(encoder.config.hidden_size * 2, eps=1e-12),
            nn.Linear(encoder.config.hidden_size * 2, 4)
        )
        self.linear=nn.Linear(encoder.config.hidden_size * 2,encoder.config.hidden_size,bias=False)
        self.sig = nn.Sigmoid()
        self.wo_pred_loss=wo_pred_loss

    def forward(self,batch):
        rank_scores_golden=self.golden(batch)
        sequence_output,rank_scores=self.encoder(batch)
        logits_gen,pooled_logits,pooled_hidden=self.decoder(batch,sequence_output)
        mse_loss_0=self.calc_kl_div(rank_scores,rank_scores_golden.detach())
        mse_loss_1 = self.calc_kl_div(pooled_logits, rank_scores_golden.detach())
        a_target=batch['a_targets']
        r_labels=batch['r_labels']
        if self.wo_pred_loss:
            cep_golden_loss = self.criterion(rank_scores_golden, a_target)
            mle_loss = self.criterion_gen(logits_gen.reshape(logits_gen.size(0) * logits_gen.size(1), -1),
                                          r_labels.view(-1))
            return mse_loss_0, mse_loss_1, cep_golden_loss, mle_loss
        else:
            cep_loss=self.criterion(rank_scores,a_target)
            cep_golden_loss=self.criterion(rank_scores_golden,a_target)
            ceg_loss=self.criterion(pooled_logits,a_target)
            mle_loss=self.criterion_gen(logits_gen.reshape(logits_gen.size(0)*logits_gen.size(1),-1),r_labels.view(-1))
            return mse_loss_0,mse_loss_1,cep_loss,cep_golden_loss,ceg_loss,mle_loss

    def predict(self,batch):
        rank_scores_golden=self.golden(batch)
        sequence_output,rank_scores=self.encoder(batch)
        outputs,pooled_hidden=self.decoder.predict(batch,sequence_output)
        outputs_a = rank_scores.max(dim=-1)[1]
        outputs_g = rank_scores_golden.max(dim=-1)[1]
        return outputs,outputs_a,outputs_g

    def batch_predict_beam(self,batch):
        sequence_output=self.encoder(batch)
        outputs=self.decoder.batch_predict_beam(batch,sequence_output)

        encoder_pooled_hidden = torch.zeros((sequence_output.size(0), sequence_output.size(-1)),
                                            dtype=sequence_output.dtype).cuda()
        attn_mask = batch['attn_masks']
        for i in range(sequence_output.size(0)):
            len = sum(attn_mask[i])
            tmp = torch.mean(sequence_output[i, :len, :], dim=0)
            encoder_pooled_hidden[i] = tmp
        return outputs

    def calc_kl_div(self,mrc_outputs, explanation_outputs, temperature=1.0):
        loss_kl = F.kl_div(
            input=F.log_softmax(mrc_outputs / temperature, dim=-1),
            target=F.softmax(explanation_outputs / temperature, dim=-1),
            reduction="batchmean",
        ) * (temperature ** 2)

        return loss_kl

class Graph_pretrain_UniterForVisualCommonsenseReasoning(UniterPreTrainedModel):
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

        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']

        vis_edge=batch['vis_edge']


        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        attn_masks = batch['attn_masks']
        gather_index = batch['gather_index']
        txt_type_ids = batch['txt_type_ids']
        input_region=batch['input_region']
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attn_masks, gather_index,
                                      output_all_encoded_layers=False,
                                      txt_type_ids=txt_type_ids)
        pooled_output = self.uniter.pooler(sequence_output)

        rank_scores = self.vcr_output(pooled_output)

        return sequence_output,rank_scores

class Graph_pretrain_UNITER_GPT(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 ):
        super(Graph_pretrain_UNITER_GPT, self).__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.criterion_gen=torch.nn.CrossEntropyLoss(reduction='mean',ignore_index=0)
        self.a_output = nn.Sequential(
            nn.Linear(encoder.config.hidden_size, encoder.config.hidden_size * 2),
            nn.ReLU(),
            LayerNorm(encoder.config.hidden_size * 2, eps=1e-12),
            nn.Linear(encoder.config.hidden_size * 2, 4)
        )
        self.linear=nn.Linear(encoder.config.hidden_size * 2,encoder.config.hidden_size,bias=False)
        self.sig = nn.Sigmoid()

    def forward(self,batch,rank_scores_golden):
        sequence_output,rank_scores=self.encoder(batch)
        logits_gen,pooled_logits,pooled_hidden=self.decoder(batch,sequence_output)

        mse_loss_0=self.calc_kl_div(rank_scores,rank_scores_golden.detach())
        mse_loss_1 = self.calc_kl_div(pooled_logits, rank_scores_golden.detach())
        a_target=batch['a_targets']
        r_labels=batch['r_labels']
        cep_loss=self.criterion(rank_scores,a_target)

        ceg_loss=self.criterion(pooled_logits,a_target)
        mle_loss=self.criterion_gen(logits_gen.reshape(logits_gen.size(0)*logits_gen.size(1),-1),r_labels.view(-1))
        return mse_loss_0,mse_loss_1,cep_loss,ceg_loss,mle_loss

    def predict(self,batch,rank_scores_golden):
        sequence_output,rank_scores=self.encoder(batch)
        outputs,pooled_hidden=self.decoder.predict(batch,sequence_output)
        outputs_a = rank_scores.max(dim=-1)[1]
        outputs_g = rank_scores_golden.max(dim=-1)[1]
        return outputs,outputs_a,outputs_g

    def batch_predict_beam(self,batch):
        sequence_output=self.encoder(batch)
        outputs=self.decoder.batch_predict_beam(batch,sequence_output)

        encoder_pooled_hidden = torch.zeros((sequence_output.size(0), sequence_output.size(-1)),
                                            dtype=sequence_output.dtype).cuda()
        attn_mask = batch['attn_masks']
        for i in range(sequence_output.size(0)):
            len = sum(attn_mask[i])
            tmp = torch.mean(sequence_output[i, :len, :], dim=0)
            encoder_pooled_hidden[i] = tmp
        return outputs

    def calc_kl_div(self,mrc_outputs, explanation_outputs, temperature=1.0):
        loss_kl = F.kl_div(
            input=F.log_softmax(mrc_outputs / temperature, dim=-1),
            target=F.softmax(explanation_outputs / temperature, dim=-1),
            reduction="batchmean",
        ) * (temperature ** 2)

        return loss_kl
