from collections import defaultdict
import copy
from torch import nn
from torch.nn import functional as F
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
from torch.nn import CrossEntropyLoss
import torch
import numpy as np
import warnings
# from .layer import GELU
from transformers import T5Tokenizer
from transformers.models.t5.modeling_t5 import T5PreTrainedModel,T5Stack,T5_INPUTS_DOCSTRING,T5ForConditionalGeneration
from .model import (
    UniterPreTrainedModel, UniterModel)
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from transformers.models.t5.configuration_t5 import T5Config
from transformers.file_utils import     (ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.utils import logging
logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = "T5Config"
_TOKENIZER_FOR_DOC = "T5Tokenizer"
# class T5ForConditionalGeneration(T5PreTrainedModel):
#     _keys_to_ignore_on_load_missing = [
#         r"encoder\.embed_tokens\.weight",
#         r"decoder\.embed_tokens\.weight",
#         r"lm_head\.weight",
#     ]
#     _keys_to_ignore_on_load_unexpected = [
#         r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
#     ]
#
#     def __init__(self, config):
#         super().__init__(config)
#         self.model_dim = config.d_model
#
#         self.shared = nn.Embedding(config.vocab_size, config.d_model)
#
#         encoder_config = copy.deepcopy(config)
#         encoder_config.is_decoder = False
#         encoder_config.use_cache = False
#         encoder_config.is_encoder_decoder = False
#         self.encoder = T5Stack(encoder_config, self.shared)
#
#         decoder_config = copy.deepcopy(config)
#         decoder_config.is_decoder = True
#         decoder_config.is_encoder_decoder = False
#         decoder_config.num_layers = config.num_decoder_layers
#         self.decoder = T5Stack(decoder_config, self.shared)
#
#         self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
#
#         self.init_weights()
#
#         # Model parallel
#         self.model_parallel = False
#         self.device_map = None
#
#     # @add_start_docstrings(PARALLELIZE_DOCSTRING)
#     # def parallelize(self, device_map=None):
#     #     self.device_map = (
#     #         get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
#     #         if device_map is None
#     #         else device_map
#     #     )
#     #     assert_device_map(self.device_map, len(self.encoder.block))
#     #     self.encoder.parallelize(self.device_map)
#     #     self.decoder.parallelize(self.device_map)
#     #     self.lm_head = self.lm_head.to(self.decoder.first_device)
#     #     self.model_parallel = True
#     #
#     # @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
#     # def deparallelize(self):
#     #     self.encoder.deparallelize()
#     #     self.decoder.deparallelize()
#     #     self.encoder = self.encoder.to("cpu")
#     #     self.decoder = self.decoder.to("cpu")
#     #     self.lm_head = self.lm_head.to("cpu")
#     #     self.model_parallel = False
#     #     self.device_map = None
#     #     torch.cuda.empty_cache()
#
#     def get_input_embeddings(self):
#         return self.shared
#
#     def set_input_embeddings(self, new_embeddings):
#         self.shared = new_embeddings
#         self.encoder.set_input_embeddings(new_embeddings)
#         self.decoder.set_input_embeddings(new_embeddings)
#
#     def set_output_embeddings(self, new_embeddings):
#         self.lm_head = new_embeddings
#
#     def get_output_embeddings(self):
#         return self.lm_head
#
#     def get_encoder(self):
#         return self.encoder
#
#     def get_decoder(self):
#         return self.decoder
#
#     @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
#     @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         decoder_input_ids=None,
#         decoder_attention_mask=None,
#         head_mask=None,
#         decoder_head_mask=None,
#         encoder_outputs=None,
#         past_key_values=None,
#         inputs_embeds=None,
#         decoder_inputs_embeds=None,
#         labels=None,
#         use_cache=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#     ):
#
#         use_cache = use_cache if use_cache is not None else self.config.use_cache
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#
#         # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
#         if head_mask is not None and decoder_head_mask is None:
#             if self.config.num_layers == self.config.num_decoder_layers:
#                 decoder_head_mask = head_mask
#
#         # Encode if needed (training, first prediction pass)
#         # if encoder_outputs is None:
#         #     # Convert encoder inputs in embeddings if needed
#         #     encoder_outputs = self.encoder(
#         #         input_ids=input_ids,
#         #         attention_mask=attention_mask,
#         #         inputs_embeds=inputs_embeds,
#         #         head_mask=head_mask,
#         #         output_attentions=output_attentions,
#         #         output_hidden_states=output_hidden_states,
#         #         return_dict=return_dict,
#         #     )
#         # elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
#         #     encoder_outputs = BaseModelOutput(
#         #         last_hidden_state=encoder_outputs[0],
#         #         hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
#         #         attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
#         #     )
#
#         hidden_states = encoder_outputs[0]
#
#         if self.model_parallel:
#             torch.cuda.set_device(self.decoder.first_device)
#
#         if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
#             # get decoder inputs from shifting lm labels to the right
#             decoder_input_ids = self._shift_right(labels)
#
#         # If decoding with past key value states, only the last tokens
#         # should be given as an input
#         if past_key_values is not None:
#             assert labels is None, "Decoder should not use cached key value states when training."
#             if decoder_input_ids is not None:
#                 decoder_input_ids = decoder_input_ids[:, -1:]
#             if decoder_inputs_embeds is not None:
#                 decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]
#
#         # Set device for model parallelism
#         if self.model_parallel:
#             torch.cuda.set_device(self.decoder.first_device)
#             hidden_states = hidden_states.to(self.decoder.first_device)
#             if decoder_input_ids is not None:
#                 decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
#             if attention_mask is not None:
#                 attention_mask = attention_mask.to(self.decoder.first_device)
#             if decoder_attention_mask is not None:
#                 decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)
#
#         # Decode
#         decoder_outputs = self.decoder(
#             input_ids=decoder_input_ids,
#             attention_mask=decoder_attention_mask,
#             inputs_embeds=decoder_inputs_embeds,
#             past_key_values=past_key_values,
#             encoder_hidden_states=hidden_states,
#             encoder_attention_mask=attention_mask,
#             head_mask=decoder_head_mask,
#             encoder_head_mask=head_mask,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#
#         sequence_output = decoder_outputs[0]
#
#         # Set device for model parallelism
#         if self.model_parallel:
#             torch.cuda.set_device(self.encoder.first_device)
#             self.lm_head = self.lm_head.to(self.encoder.first_device)
#             sequence_output = sequence_output.to(self.lm_head.weight.device)
#
#         if self.config.tie_word_embeddings:
#             # Rescale output before projecting on vocab
#             # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
#             sequence_output = sequence_output * (self.model_dim ** -0.5)
#
#         lm_logits = self.lm_head(sequence_output)
#
#         loss = None
#         if labels is not None:
#             loss_fct = CrossEntropyLoss(ignore_index=-100)
#             loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
#             # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
#
#         if not return_dict:
#             output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
#             return ((loss,) + output) if loss is not None else output
#
#         return Seq2SeqLMOutput(
#             loss=loss,
#             logits=lm_logits,
#             past_key_values=decoder_outputs.past_key_values,
#             decoder_hidden_states=decoder_outputs.hidden_states,
#             decoder_attentions=decoder_outputs.attentions,
#             cross_attentions=decoder_outputs.cross_attentions,
#             encoder_last_hidden_state=encoder_outputs.last_hidden_state,
#             encoder_hidden_states=encoder_outputs.hidden_states,
#             encoder_attentions=encoder_outputs.attentions,
#         )
#
#     def prepare_inputs_for_generation(
#         self, input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
#     ):
#
#         # cut decoder_input_ids if past is used
#         if past is not None:
#             input_ids = input_ids[:, -1:]
#
#         return {
#             "decoder_input_ids": input_ids,
#             "past_key_values": past,
#             "encoder_outputs": encoder_outputs,
#             "attention_mask": attention_mask,
#             "use_cache": use_cache,
#         }
#
#     def _reorder_cache(self, past, beam_idx):
#         # if decoder past is not included in output
#         # speedy decoding is disabled and no need to reorder
#         if past is None:
#             logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
#             return past
#
#         reordered_decoder_past = ()
#         for layer_past_states in past:
#             # get the correct batch idx from layer past batch dim
#             # batch dim of `past` is at 2nd position
#             reordered_layer_past_states = ()
#             for layer_past_state in layer_past_states:
#                 # need to set correct `past` for each of the four key / value states
#                 reordered_layer_past_states = reordered_layer_past_states + (
#                     layer_past_state.index_select(0, beam_idx),
#                 )
#
#             assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
#             assert len(reordered_layer_past_states) == len(layer_past_states)
#
#             reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
#         return reordered_decoder_past

class QR_T5_UniterForVisualCommonsenseReasoning(UniterPreTrainedModel):
    """ Finetune UNITER for VCR
    """
    def __init__(self, config, img_dim):
        super().__init__(config, img_dim)
        self.uniter = UniterModel(config, img_dim)
        self.tokenizer=T5Tokenizer.from_pretrained('/raid/yq/UNITER/pretrain/t5_base')
        self.tokenizer.add_special_tokens({'bos_token':'<s>'})
        self.t5=T5ForConditionalGeneration.from_pretrained('/raid/yq/UNITER/pretrain/t5_base')
        #用于将encoder输入转为decoder
        self.project = nn.Linear(768, 29077)
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


        trgs=batch['targets']
        target =self.tokenizer.batch_encode_plus(trgs,padding=True,return_attention_mask=True,return_tensors='pt')
        trg_ids=target.data['input_ids'].cuda()
        attention_mask=target.data['attention_mask'].cuda()
        labels=trg_ids[:,1:]
        pad = torch.zeros((labels.size(0), 1), dtype=torch.int64).cuda()
        labels = torch.cat((labels, pad), dim=1)
        out=self.t5(encoder_outputs=[sequence_output],attention_mask=attn_masks,decoder_input_ids=trg_ids,decoder_attention_mask=attention_mask,labels=labels)
        return out.logits,labels

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

        sequence_output = self.linear(sequence_output)

        trgs = batch['targets']
        target = self.tokenizer.batch_encode_plus(trgs, padding=True, return_attention_mask=True, return_tensors='pt')
        trg_ids = target.data['input_ids'].cuda()
        labels = trg_ids[:, 1:]
        pad = torch.zeros((labels.size(0), 1), dtype=torch.int64).cuda()
        labels = torch.cat((labels, pad), dim=1)

        predict_his = torch.zeros(trg_ids.size(), dtype=int).cuda()
        outputs=torch.zeros(trg_ids.size(), dtype=int).cuda()
        predict_his[:, 0] =trg_ids[:, 0]


        for index in range(0, trg_ids.size(1)-1):
            out = self.t5(encoder_outputs=[sequence_output], attention_mask=attn_masks, decoder_input_ids=predict_his[:, :index + 1])
            # past_key_values=out.past_key_values
            out = out.logits[:,-1:,:]
            label = self.choose_top_word(out)
            for i in range(trg_ids.size(0)):
                predict_his[i, index + 1] = label[i]
                outputs[i, index] = label[i]
        return outputs,labels

    def choose_top_word(self, prob):
        softmax = nn.Softmax(dim=-1)
        prob_soft = softmax(prob)
        prob_soft = prob_soft.cpu().numpy()
        label = np.argmax(prob_soft, axis=2)
        label.resize(prob.size(0))
        label = torch.from_numpy(label)
        return label