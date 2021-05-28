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
from transformers.file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers import GPT2Tokenizer

class QR_GPT_UniterForVisualCommonsenseReasoning(UniterPreTrainedModel):
    """ Finetune UNITER for VCR
    """
    def __init__(self, config, img_dim):
        super().__init__(config, img_dim)
        self.uniter = UniterModel(config, img_dim)
        self.config_across = GPT2Config(vocab_size=29077, n_positions=512, n_ctx=512,
                                        n_layer=4,
                                        n_embd=1024, add_cross_attention=True, n_head=8)
        self.tokenizer=GPT2Tokenizer.from_pretrained('/raid/yq/UNITER/pretrain/gpt')
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.GPT_decoder = GPT2LMHeadModel.from_pretrained('/raid/yq/UNITER/pretrain/gpt')
        #self.linear=nn.Linear(config.hidden_size,256)
        #用于将encoder输入转为decoder
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

        #encoder_outputs=self.linear(sequence_output)

        # trg=batch['targets']
        # trg_length=batch['trg_length']
        #
        # word_embed=self.uniter.embeddings.word_embeddings(trg)
        # attention_mask = torch.zeros(trg.size(), dtype=torch.int).cuda()
        # for batch_index in range(trg.size(0)):
        #     for sent_len in range(trg_length[batch_index]):
        #         attention_mask[batch_index, sent_len] = 1
        # out=self.GPT_decoder(inputs_embeds=word_embed,encoder_hidden_states=sequence_output,attention_mask=attention_mask,encoder_attention_mask=attn_masks)

        trgs=batch['targets']
        target = self.tokenizer.batch_encode_plus(trgs, padding=True, return_attention_mask=True, return_tensors='pt')
        input_ids=target.data['input_ids'].cuda()
        attention_mask=target.data['attention_mask'].cuda()
        labels=input_ids[:,1:]
        pad = torch.zeros((labels.size(0), 1), dtype=torch.int64).cuda()
        labels = torch.cat((labels, pad), dim=1)

        out=self.GPT_decoder(input_ids=input_ids,encoder_hidden_states=sequence_output,attention_mask=attention_mask,encoder_attention_mask=attn_masks)

        out = out.logits


        return out,labels

    def predict(self,batch):
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

        trgs = batch['targets']
        target = self.tokenizer.batch_encode_plus(trgs, padding=True, return_attention_mask=True, return_tensors='pt')
        input_ids = target.data['input_ids'].cuda()
        labels = input_ids[:, 1:]

        predict_his = torch.zeros(input_ids.size(), dtype=int).cuda()
        outputs=torch.zeros(input_ids.size(), dtype=int).cuda()
        predict_his[:, 0] =input_ids[:, 0]

        for index in range(0, input_ids.size(1)-1):

            out=self.GPT_decoder(input_ids=predict_his[:, :index + 1],encoder_hidden_states=sequence_output,encoder_attention_mask=attn_masks)
            # past_key_values=out.past_key_values
            out = out.logits
            label = self.choose_top_word(out)
            for i in range(input_ids.size(0)):
                predict_his[i, index + 1] = label[i]
                outputs[i, index] = label[i]
        return outputs,labels

    def choose_top_word(self, prob):
        label = np.argmax(prob.cpu().numpy(), axis=2)
        label.resize(prob.size(0))
        label = torch.from_numpy(label)
        return label

class QR_GPT_wo_UniterForVisualCommonsenseReasoning(UniterPreTrainedModel):
    """ Finetune UNITER for VCR
    """
    def __init__(self, config, img_dim):
        super().__init__(config, img_dim)
        self.uniter = UniterModel(config, img_dim)
        self.config_across = GPT2Config(vocab_size=29077, n_positions=512, n_ctx=512,
                                        n_layer=8,
                                        n_embd=1024, add_cross_attention=True, n_head=8)
        self.GPT_decoder = GPT2Model(self.config_across)
        #self.linear=nn.Linear(config.hidden_size,256)
        #用于将encoder输入转为decoder
        self.vocab_size=29077
        self.project = nn.Linear(1024, 29077)
        self.apply(self.init_weights)
        self.max_len=80
        self.beam_size=3


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

        #encoder_outputs=self.linear(sequence_output)

        trg=batch['targets']
        trg_length=batch['trg_length']
        word_embed=self.uniter.embeddings.word_embeddings(trg)
        attention_mask = torch.zeros(trg.size(), dtype=torch.int).cuda()
        for batch_index in range(trg.size(0)):
            for sent_len in range(trg_length[batch_index]):
                attention_mask[batch_index, sent_len] = 1

        out=self.GPT_decoder(inputs_embeds=word_embed,encoder_hidden_states=sequence_output,attention_mask=attention_mask,encoder_attention_mask=attn_masks)
        out = out.last_hidden_state
        le, mb, hd = out.shape
        out = out.view(le * mb, hd)
        out = self.project(out)
        out = out.view(le, mb, -1)

        return out

    def predict(self,batch):
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

        trg = batch['targets']
        predict_his = torch.zeros((trg.size(0),self.max_len), dtype=int).cuda()
        outputs=torch.zeros((trg.size(0),self.max_len), dtype=int).cuda()
        predict_his[:, 0] =trg[:, 0]

        for index in range(0, self.max_len-1):
            word_embed = self.uniter.embeddings.word_embeddings(predict_his[:, :index + 1])
            out = self.GPT_decoder(inputs_embeds=word_embed, encoder_hidden_states=sequence_output,encoder_attention_mask=attn_masks)
            out = out.last_hidden_state
            out = out[:, -1:, :]
            le, mb, hd = out.shape
            out = out.view(le * mb, hd)
            out = self.project(out)
            out = out.view(le, mb, -1)
            label = self.choose_top_word(out)
            for i in range(trg.size(0)):
                predict_his[i, index + 1] = label[i]
                outputs[i, index] = label[i]
        return outputs

    def choose_top_word(self, prob):
        label = np.argmax(prob.cpu().numpy(), axis=2)
        label.resize(prob.size(0))
        label = torch.from_numpy(label)
        return label

    def predict_beam(self, batch):
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

        trg = batch['targets']

        beam_width = self.beam_size
        topk = 1  # how many sentence do you want to generate
        decoded_batch = []

        # decoding goes sentence by sentence
        for idx in range(trg.size(0)):

            encoder_output = sequence_output[idx, :, :].unsqueeze(0)

            # Start with the start of the sentence token

            # Number of sentence to generate
            endnodes = []
            number_required = min((topk + 1), topk - len(endnodes))
            predict_his = torch.zeros((1, self.max_len), dtype=int).cuda()
            predict_his[:,0] = 101

            # starting node -  hidden vector, previous node, word id, logp, length
            node = BeamSearchNode( None, predict_his[:,0], 0, 1)
            nodes = PriorityQueue()

            # start the queue
            nodes.put((-node.eval(), node))
            qsize = 1


            # start beam search
            while True:

                # give up when decoding takes too long
                if qsize > 2000: break


                # fetch the best node
                score, n = nodes.get()
                if n.leng==self.max_len:
                    break

                if n.wordid.item() == 102 and n.prevNode != None:
                    endnodes.append((score, n))
                    # if we reached maximum # of sentences required
                    if len(endnodes) >= number_required:
                        break
                    else:
                        continue

                # output维度为 [batch_size, vocab_size]
                # hidden维度为 [num_layers * num_directions, batch_size, hidden_size]
                # decode for one step using decoder
                node_input=n
                while node_input.prevNode:
                    predict_his[:, node_input.leng - 1] = node_input.wordid
                    node_input=node_input.prevNode

                word_embed = self.uniter.embeddings.word_embeddings(predict_his[:,:n.leng])
                out = self.GPT_decoder(inputs_embeds=word_embed, encoder_hidden_states=encoder_output,
                                       encoder_attention_mask=attn_masks[idx,:].unsqueeze(0))
                out = out.last_hidden_state
                out = out[:, -1:, :]
                le, mb, hd = out.shape
                out = out.view(le * mb, hd)
                out = self.project(out)
                out = out.view(le, mb, -1)

                # PUT HERE REAL BEAM SEARCH OF TOP
                # log_prov, indexes维度为 [batch_size, beam_width] = [1, beam_width]
                log_prob, indexes = torch.topk(out.squeeze(1), beam_width, dim=1)
                nextnodes = []

                for new_k in range(beam_width):
                    # decoded_t: [1,1],通过view(1,-1)将数字tensor变为维度为[1,1]的tensor
                    decoded_t = indexes[0][new_k].view(1, -1)
                    # log_p, int
                    log_p = log_prob[0][new_k].item()  # item()将tensor数字变为int

                    node = BeamSearchNode( n, decoded_t, n.logp + log_p, n.leng + 1)
                    score = -node.eval()
                    nextnodes.append((score, node))

                # put them into queue
                for i in range(len(nextnodes)):
                    score, nn = nextnodes[i]
                    nodes.put((score, nn))
                    # increase qsize
                qsize += len(nextnodes) - 1

            # choose nbest paths, back trace them
            if len(endnodes) == 0:
                endnodes = [nodes.get() for _ in range(topk)]

            utterances = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterance = []
                utterance.append(n.wordid.item())
                # back trace
                while n.prevNode != None:
                    n = n.prevNode
                    utterance.append(n.wordid.item())

                utterance = utterance[::-1]
                utterances.append(utterance)

            decoded_batch.append(utterances)

        return decoded_batch

    def batch_predict_beam(self,batch):
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

        trg = batch['targets']
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
            word_embed = self.uniter.embeddings.word_embeddings(input_ids)
            out = self.GPT_decoder(inputs_embeds=word_embed, encoder_hidden_states=sequence_output_ex,
                                   encoder_attention_mask=attn_masks_ex)
            out = out.last_hidden_state
            out = out[:, -1:, :]
            out = self.project(out)
            out=out.squeeze(1)
            scores = F.log_softmax(out, dim=-1)  # log_softmax
            next_scores = scores + beam_scores[:, None].expand_as(scores).cuda()  # 累加上以前的scores
            next_scores = next_scores.view(
                batch_size, self.beam_size * self.vocab_size
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
                    beam_id = beam_token_id // self.vocab_size  # 1
                    token_id = beam_token_id % self.vocab_size  # 1
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

class GPT_pretrain(nn.Module):
    def __init__(self,
                 d_model=512,
                 num_layer_0=4,
                 dropout=0.2,
                 embed_dim=768,
                 h=8,
                 d_ff=2048
                 ):
        super(GPT_pretrain, self).__init__()
        self.w2i,self.i2w = make_dict()
        self.vocab_num = len(self.w2i.keys())
        self.embedding = nn.Embedding(self.vocab_num, embed_dim)
        self.dropout = dropout
        self.config_across = GPT2Config(vocab_size=29077, n_positions=512, n_ctx=512,
                                        n_layer=8,
                                        n_embd=768, add_cross_attention=True, n_head=8)
        self.GPT_decoder = GPT2Model(self.config_across)
        self.project = nn.Linear(embed_dim, self.vocab_num)

        for par in self.GPT_decoder.parameters():
            if par.ndim > 2:
                xavier_normal_(par)

    def forward(self, batch):
        x=batch['targets']
        x_lengths=batch['trg_length']
        word_embed = self.embedding(x)
        attention_mask = torch.zeros(x.size(), dtype=torch.int).cuda()
        for batch_index in range(x.size(0)):
            for sent_len in range(x_lengths[batch_index]):
                attention_mask[batch_index, sent_len] = 1
        out=self.GPT_decoder(inputs_embeds=word_embed,attention_mask=attention_mask)
        out = out.last_hidden_state
        # le, mb, hd = out.shape
        # out = out.view(le * mb, hd)
        out = self.project(out)
        # out = out.view(le, mb, -1)
        return out

class Encoder_UniterForVisualCommonsenseReasoning(UniterPreTrainedModel):
    """ 只有Encoder
    """
    def __init__(self, config, img_dim):
        super().__init__(config, img_dim)
        self.uniter = UniterModel(config, img_dim)
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


        return sequence_output,attn_masks

class Encoder_add_UniterForVisualCommonsenseReasoning(UniterPreTrainedModel):
    """ 文本表示部分融合gt特征
    """
    def __init__(self, config, img_dim):
        super().__init__(config, img_dim)
        self.uniter = UniterModel_add(config, img_dim)
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
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attn_masks = batch['attn_masks']
        gather_index = batch['gather_index']
        txt_type_ids = batch['txt_type_ids']
        input_region=batch['input_region']
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attn_masks, gather_index,
                                      output_all_encoded_layers=False,
                                      txt_type_ids=txt_type_ids,
                                      input_region=input_region)


        return sequence_output,attn_masks

class Encoder_graph_UniterForVisualCommonsenseReasoning(UniterPreTrainedModel):
    """ 文本表示部分融合gt特征
    """
    def __init__(self, config, img_dim):
        super().__init__(config, img_dim)
        self.uniter = UniterModel_graph(config, img_dim)
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


        return sequence_output,attn_masks

class Encoder_GRU(UniterPreTrainedModel):
    """ 只有Encoder
    """
    def __init__(self, config, img_dim):
        super().__init__(config, img_dim)
        self.enoder = GRU_encoder(config, img_dim)
        self.apply(self.init_weights)

    def init_type_embedding(self):
        new_emb = nn.Embedding(4, self.enoder.config.hidden_size)
        new_emb.apply(self.init_weights)
        for i in [0, 1]:
            emb = self.enoder.embeddings.token_type_embeddings.weight.data[i, :]
            new_emb.weight.data[i, :].copy_(emb)
        emb = self.enoder.embeddings.token_type_embeddings.weight.data[0, :]
        new_emb.weight.data[2, :].copy_(emb)
        new_emb.weight.data[3, :].copy_(emb)
        self.enoder.embeddings.token_type_embeddings = new_emb

    def init_word_embedding(self, num_special_tokens):
        orig_word_num = self.enoder.embeddings.word_embeddings.weight.size(0)
        new_emb = nn.Embedding(
            orig_word_num + num_special_tokens, self.enoder.config.hidden_size)
        new_emb.apply(self.init_weights)
        emb = self.enoder.embeddings.word_embeddings.weight.data
        new_emb.weight.data[:orig_word_num, :].copy_(emb)
        self.enoder.embeddings.word_embeddings = new_emb


    def forward(self, batch):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attn_masks = batch['attn_masks']
        gather_index = batch['gather_index']
        txt_type_ids = batch['txt_type_ids']
        encoder_output,encoder_hidden = self.enoder(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attn_masks, gather_index,
                                      output_all_encoded_layers=False,
                                      txt_type_ids=txt_type_ids)


        return encoder_output,encoder_hidden

class Decodder_GRU(nn.Module):
    def __init__(self,
                 hidden_size=512,
                 dropout=0.2,
                 embed_dim=1024,
                 ):
        super(Decodder_GRU, self).__init__()
        self.w2i,self.i2w = make_dict()
        self.vocab_num = len(self.w2i.keys())
        self.embedding = nn.Embedding(self.vocab_num, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(embed_dim, hidden_size, 1)
        self.project = nn.Linear(hidden_size, self.vocab_num)

    def forward(self, batch,enc_out,enc_hidden):
        x = batch['targets']
        x_lengths = batch['trg_length']
        word_embed = self.embedding(x)
        word_embed=self.dropout(word_embed)
        enc_pack = rnn_utils.pack_padded_sequence(word_embed, lengths=x_lengths, batch_first=True, enforce_sorted=False)
        out, h0 = self.rnn(enc_pack,enc_hidden)  # (len, batch, hidden)
        out, _ = rnn_utils.pad_packed_sequence(sequence=out, batch_first=True, padding_value=0)
        out=self.project(out)

        return out
class BaseModel(nn.Module):
    def __init__(self,
                 encoder,
                 decoder
                 ):
        super(BaseModel, self).__init__()
        self.encoder=encoder
        self.decoder=decoder

    def forward(self,batch):
        enc_out,enc_hidden=self.encoder(batch)
        outputs=self.decoder(batch,enc_out,enc_hidden)
        return outputs

    def predict(self,batch):
        enc_out,enc_hidden = self.encoder(batch)
        outputs=self.decoder.predict(batch,enc_out,enc_hidden)
        return outputs

    def batch_predict_beam(self,batch):
        enc_hid = self.encoder(batch)
        outputs=self.decoder.batch_predict_beam(batch,enc_hid)
        return outputs

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

        for par in self.GPT_decoder.parameters():
            if par.ndim > 2:
                xavier_normal_(par)

    def forward(self, batch,enc_hid,enc_attn):
        x=batch['targets']
        x_lengths=batch['trg_length']
        word_embed = self.embedding(x)
        attention_mask = torch.zeros(x.size(), dtype=torch.int).cuda()
        for batch_index in range(x.size(0)):
            for sent_len in range(x_lengths[batch_index]):
                attention_mask[batch_index, sent_len] = 1
        out=self.GPT_decoder(inputs_embeds=word_embed,attention_mask=attention_mask,encoder_hidden_states=enc_hid,encoder_attention_mask=enc_attn)
        out = out.last_hidden_state
        # le, mb, hd = out.shape
        # out = out.view(le * mb, hd)
        out = self.project(out)
        # out = out.view(le, mb, -1)
        return out

    def predict(self,batch,enc_hid,enc_attn):

        trg = batch['targets']
        predict_his = torch.zeros((trg.size(0),self.max_len), dtype=int).cuda()
        outputs=torch.zeros((trg.size(0),self.max_len), dtype=int).cuda()
        predict_his[:, 0] =trg[:, 0]

        for index in range(0, self.max_len-1):
            word_embed = self.embedding(predict_his[:, :index + 1])
            out = self.GPT_decoder(inputs_embeds=word_embed, encoder_hidden_states=enc_hid,encoder_attention_mask=enc_attn)
            out = out.last_hidden_state
            out = out[:, -1:, :]
            # le, mb, hd = out.shape
            # out = out.view(le * mb, hd)
            out = self.project(out)
            # out = out.view(le, mb, -1)
            label = self.choose_top_word(out)
            for i in range(trg.size(0)):
                predict_his[i, index + 1] = label[i]
                outputs[i, index] = label[i]
        return outputs

    def choose_top_word(self, prob):
        label = np.argmax(prob.cpu().numpy(), axis=2)
        label.resize(prob.size(0))
        label = torch.from_numpy(label)
        return label

    def batch_predict_beam(self, batch,enc_hid,enc_attn):

        trg = batch['targets']

        beam_width = self.beam_size
        topk = 1  # how many sentence do you want to generate
        decoded_batch = []

        # decoding goes sentence by sentence
        for idx in range(trg.size(0)):

            encoder_output = enc_hid[idx, :, :].unsqueeze(0)

            # Start with the start of the sentence token

            # Number of sentence to generate
            endnodes = []
            number_required = min((topk + 1), topk - len(endnodes))
            predict_his = torch.zeros((1, self.max_len), dtype=int).cuda()
            predict_his[:,0] = 101

            # starting node -  hidden vector, previous node, word id, logp, length
            node = BeamSearchNode( None, predict_his[:,0], 0, 1)
            nodes = PriorityQueue()

            # start the queue
            nodes.put((-node.eval(), node))
            qsize = 1


            # start beam search
            while True:

                # give up when decoding takes too long
                if qsize > 2000: break


                # fetch the best node
                score, n = nodes.get()
                if n.leng==self.max_len:
                    break

                if n.wordid.item() == 102 and n.prevNode != None:
                    endnodes.append((score, n))
                    # if we reached maximum # of sentences required
                    if len(endnodes) >= number_required:
                        break
                    else:
                        continue

                # output维度为 [batch_size, vocab_size]
                # hidden维度为 [num_layers * num_directions, batch_size, hidden_size]
                # decode for one step using decoder
                node_input=n
                while node_input.prevNode:
                    predict_his[:, node_input.leng - 1] = node_input.wordid
                    node_input=node_input.prevNode

                word_embed = self.embedding(predict_his[:,:n.leng])
                out = self.GPT_decoder(inputs_embeds=word_embed, encoder_hidden_states=encoder_output,
                                       encoder_attention_mask=enc_attn[idx,:].unsqueeze(0))
                out = out.last_hidden_state
                out = out[:, -1:, :]
                # le, mb, hd = out.shape
                # out = out.view(le * mb, hd)
                out = self.project(out)
                # out = out.view(le, mb, -1)

                # PUT HERE REAL BEAM SEARCH OF TOP
                # log_prov, indexes维度为 [batch_size, beam_width] = [1, beam_width]
                log_prob, indexes = torch.topk(out.squeeze(1), beam_width, dim=1)
                nextnodes = []

                for new_k in range(beam_width):
                    # decoded_t: [1,1],通过view(1,-1)将数字tensor变为维度为[1,1]的tensor
                    decoded_t = indexes[0][new_k].view(1, -1)
                    # log_p, int
                    log_p = log_prob[0][new_k].item()  # item()将tensor数字变为int

                    node = BeamSearchNode( n, decoded_t, n.logp + log_p, n.leng + 1)
                    score = -node.eval()
                    nextnodes.append((score, node))

                # put them into queue
                for i in range(len(nextnodes)):
                    score, nn = nextnodes[i]
                    nodes.put((score, nn))
                    # increase qsize
                qsize += len(nextnodes) - 1

            # choose nbest paths, back trace them
            if len(endnodes) == 0:
                endnodes = [nodes.get() for _ in range(topk)]

            utterances = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterance = []
                utterance.append(n.wordid.item())
                # back trace
                while n.prevNode != None:
                    n = n.prevNode
                    utterance.append(n.wordid.item())

                utterance = utterance[::-1]
                utterances.append(utterance)

            decoded_batch.append(utterances)

        return decoded_batch

class GPT_decoder_base(nn.Module):
    def __init__(self,
                 d_model=512,
                 num_layer=8,
                 dropout=0.2,
                 embed_dim=1024,
                 h=8,
                 d_ff=2048
                 ):
        super(GPT_decoder_base, self).__init__()
        self.w2i,self.i2w = make_dict()
        self.vocab_num = len(self.w2i.keys())
        self.embedding = nn.Embedding(self.vocab_num, embed_dim)
        self.dropout = dropout
        self.config_across = GPT2Config(vocab_size=self.vocab_num, n_positions=d_model, n_ctx=d_model,
                                        n_layer=num_layer,
                                        n_embd=embed_dim, add_cross_attention=True, n_head=h)
        self.GPT_decoder = GPT2Model(self.config_across)
        self.linear_0=nn.Linear(768,embed_dim)
        self.project = nn.Linear(embed_dim, self.vocab_num,bias=False)
        self.max_len=100
        self.beam_size=3

        for par in self.GPT_decoder.parameters():
            if par.ndim > 2:
                xavier_normal_(par)

    def forward(self, batch,enc_hid,enc_attn):
        enc_hid=self.linear_0(enc_hid)
        x=batch['targets']
        x_lengths=batch['trg_length']
        word_embed = self.embedding(x)
        attention_mask = torch.zeros(x.size(), dtype=torch.int).cuda()
        for batch_index in range(x.size(0)):
            for sent_len in range(x_lengths[batch_index]):
                attention_mask[batch_index, sent_len] = 1
        out=self.GPT_decoder(inputs_embeds=word_embed,attention_mask=attention_mask,encoder_hidden_states=enc_hid,encoder_attention_mask=enc_attn)
        out = out.last_hidden_state
        # le, mb, hd = out.shape
        # out = out.view(le * mb, hd)
        out = self.project(out)
        # out = out.view(le, mb, -1)
        return out

    def predict(self,batch,enc_hid,enc_attn):
        enc_hid = self.linear_0(enc_hid)
        trg = batch['targets']
        predict_his = torch.zeros((trg.size(0),self.max_len), dtype=int).cuda()
        outputs=torch.zeros((trg.size(0),self.max_len), dtype=int).cuda()
        predict_his[:, 0] =trg[:, 0]

        for index in range(0, self.max_len-1):
            word_embed = self.embedding(predict_his[:, :index + 1])
            out = self.GPT_decoder(inputs_embeds=word_embed, encoder_hidden_states=enc_hid,encoder_attention_mask=enc_attn)
            out = out.last_hidden_state
            out = out[:, -1:, :]
            out = self.project(out)
            label = self.choose_top_word(out)
            for i in range(trg.size(0)):
                predict_his[i, index + 1] = label[i]
                outputs[i, index] = label[i]
        return outputs

    def choose_top_word(self, prob):
        label = np.argmax(prob.cpu().numpy(), axis=2)
        label.resize(prob.size(0))
        label = torch.from_numpy(label)
        return label

    def batch_predict_beam(self,batch,sequence_output,attn_masks):
        sequence_output=self.linear_0(sequence_output)

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

class UNITER_GPT(nn.Module):
    def __init__(self,
                 encoder,
                 decoder
                 ):
        super(UNITER_GPT, self).__init__()
        self.encoder=encoder
        self.decoder=decoder

    def forward(self,batch):
        enc_hid,enc_attn=self.encoder(batch)
        outputs=self.decoder(batch,enc_hid,enc_attn)
        return outputs

    def predict(self,batch):
        enc_hid, enc_attn = self.encoder(batch)
        outputs=self.decoder.predict(batch,enc_hid,enc_attn)
        return outputs

    def batch_predict_beam(self,batch):
        enc_hid, enc_attn = self.encoder(batch)
        outputs=self.decoder.batch_predict_beam(batch,enc_hid,enc_attn)
        return outputs

class UNITER_GPT_wo_encoder(nn.Module):
    def __init__(self,
                 encoder,
                 decoder
                 ):
        super(UNITER_GPT_wo_encoder, self).__init__()
        self.encoder=encoder
        self.decoder=decoder

    def forward(self,batch):
        with torch.no_grad():
            enc_hid,enc_attn=self.encoder(batch)
        outputs=self.decoder(batch,enc_hid,enc_attn)
        return outputs

    def predict(self,batch):
        enc_hid, enc_attn = self.encoder(batch)
        outputs=self.decoder.predict(batch,enc_hid,enc_attn)
        return outputs

    def batch_predict_beam(self,batch):
        enc_hid, enc_attn = self.encoder(batch)
        outputs=self.decoder.batch_predict_beam(batch,enc_hid,enc_attn)
        return outputs

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
