import copy
import json
import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip
from cytoolz import concat
from .data import (DetectFeatTxtTokDataset, TxtTokLmdb, DetectFeatLmdb,
                   TxtLmdb, get_ids_and_lens, pad_tensors,
                   get_gather_index)
import numpy as np
class VcrTxtTokLmdb(TxtTokLmdb):
    def __init__(self, db_dir, max_txt_len=120, task="qa,qar"):
        # assert task == "qa" or task == "qar" or task == "qa,qar",\
        #     "VCR only support the following tasks: 'qa', 'qar' or 'qa,qar'"
        self.task = task
        if task == "qa,qar":
            id2len_task = "qar"
        else:
            id2len_task = task
        if max_txt_len == -1:
            self.id2len = json.load(
                open(f'{db_dir}/id2len_{id2len_task}.json'))
        else:
            self.id2len = {
                id_: len_
                for id_, len_ in json.load(
                    open(f'{db_dir}/id2len_{id2len_task}.json')
                    ).items()
                if len_ <= max_txt_len
            }
        #self.id2len=dict_slice(self.id2len)

        self.db_dir = db_dir
        self.db = TxtLmdb(db_dir, readonly=True)
        meta = json.load(open(f'{db_dir}/meta.json', 'r'))
        self.cls_ = meta['CLS']
        self.sep = meta['SEP']
        self.mask = meta['MASK']
        self.v_range = meta['v_range']

class VcrDetectFeatTxtTokDataset(DetectFeatTxtTokDataset):
    def __init__(self, txt_db, img_db_gt=None, img_db=None):
        # assert not (img_db_gt is None and img_db is None),\
        #     "img_db_gt and img_db cannot all be None"
        # assert isinstance(txt_db, VcrTxtTokLmdb)
        assert img_db_gt is None or isinstance(img_db_gt, DetectFeatLmdb)
        assert img_db is None or isinstance(img_db, DetectFeatLmdb)
        self.txt_db = txt_db
        self.img_db = img_db
        self.img_db_gt = img_db_gt
        self.ls = img_db_gt
        self.task = self.txt_db.task
        txt_lens, self.ids = get_ids_and_lens(txt_db)

        txt2img = txt_db.txt2img

        if self.img_db and self.img_db_gt:
            self.lens = [tl+self.img_db_gt.name2nbb[txt2img[id_][0]] +
                         self.img_db.name2nbb[txt2img[id_][1]]
                         for tl, id_ in zip(txt_lens, self.ids)]
        elif self.img_db:
            self.lens = [tl+self.img_db.name2nbb[txt2img[id_][1]]
                         for tl, id_ in zip(txt_lens, self.ids)]
        elif self.img_db_gt:
            self.lens = [tl+self.img_db_gt.name2nbb[txt2img[id_][0]]
                         for tl, id_ in zip(txt_lens, self.ids)]
        else:
            self.lens = [tl  for tl in txt_lens]

    def _get_img_feat(self, fname_gt, fname):
        if self.img_db and self.img_db_gt:
            img_feat_gt, bb_gt = self.img_db_gt[fname_gt]
            img_bb_gt = torch.cat([bb_gt, bb_gt[:, 4:5]*bb_gt[:, 5:]], dim=-1)

            img_feat, bb = self.img_db[fname]
            img_bb = torch.cat([bb, bb[:, 4:5]*bb[:, 5:]], dim=-1)

            img_feat = torch.cat([img_feat_gt, img_feat], dim=0)
            img_bb = torch.cat([img_bb_gt, img_bb], dim=0)
            num_bb = img_feat.size(0)
        elif self.img_db:
            img_feat, bb = self.img_db[fname]
            img_bb = torch.cat([bb, bb[:, 4:5]*bb[:, 5:]], dim=-1)
            num_bb = img_feat.size(0)
        elif self.img_db_gt:
            img_feat, bb = self.img_db_gt[fname_gt]
            img_bb = torch.cat([bb, bb[:, 4:5]*bb[:, 5:]], dim=-1)
            num_bb = img_feat.size(0)
        return img_feat, img_bb, num_bb

def move_to_cuda(batch):
    if isinstance(batch, torch.Tensor):
        return batch.cuda(non_blocking=True)
    elif isinstance(batch, list):
        new_batch = [move_to_cuda(t) for t in batch]
    elif isinstance(batch, tuple):
        new_batch = tuple(move_to_cuda(t) for t in batch)
    elif isinstance(batch, dict):
        new_batch = {n: move_to_cuda(t) for n, t in batch.items()}
    else:
        return batch
    return new_batch

class Comet_qar_val_VcrDataset(VcrDetectFeatTxtTokDataset):
    def __init__(self,split, tokenizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # assert self.task != "qa,qar",\
        #     "loading training dataset with each task separately"
        self.split=split
        comet_path = "/raid/yq/visualcompet/experiments/image-inference-80000-ckpt/test_sample_1_num_5_top_k_0_top_p_0.9_ids.json"
        self.caption = self.read_comet(comet_path)
        self.tokenizer = tokenizer
        self.max_cap = 100

    def read_comet(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        caption_dict = {}
        for line in data:
            if line['inference_relation'] != 'before':
                key = line['img_fn']
                if key in caption_dict.keys():
                    caption_dict[key].append(line)
                else:
                    caption_dict[key] = []
                    caption_dict[key].append(line)
        return caption_dict

    def bert_tokenize(self, dict, region):
        ids = []
        region_index = np.nonzero(region)[0]
        redion_ids = [region[i] for i in region_index]
        for dic in dict:
            region_in = False
            for id in redion_ids:
                if str(id) == dic['event'].split()[0]:
                    region_in = True
                    break
            if region_in == True and id <= 81:
                text = dic['generations_ids']
                for sent in text:
                    # 原句中没有主语,主语就是region_id
                    ws_id = 28995 + id
                    ids.extend([ws_id])
                    ids.extend(sent)

                    ids.extend(self.tokenizer.convert_tokens_to_ids(['.']))
        return ids

    def _get_input_ids(self, txt_dump):
        # text input
        input_ids_q = txt_dump['input_ids']
        type_ids_q = [0] * len(input_ids_q)

        # 增加caption作为解释
        try:
            caption = self.caption[txt_dump['img_fn']]
            caption_id = self.bert_tokenize(caption, txt_dump['toked_txt_region_tokens'])
            caption_id = caption_id[:self.max_cap]
            input_ids_q = input_ids_q + [self.txt_db.sep] + copy.deepcopy(caption_id)
            type_ids_q = type_ids_q + [3] * (len(caption_id) + 1)
        except:
            caption = None

        input_ids_as = txt_dump['input_ids_as']
        input_ids_qa=[]
        type_ids_qa=[]
        for input_ids_a in input_ids_as:
            input_ids_qa_tmp = input_ids_q + [self.txt_db.sep] + input_ids_a
            type_ids_qa_tmp = type_ids_q + [2] * (len(input_ids_a) + 1)
            input_ids_qa.append(input_ids_qa_tmp)
            type_ids_qa.append(type_ids_qa_tmp)
        return input_ids_q, type_ids_q,input_ids_qa,type_ids_qa

    def __getitem__(self, i):
        """
        [[txt, img1],
         [txt, img2]]
        """

        example = super().__getitem__(i)
        qid = self.ids[i]
        img_feat, img_pos_feat, num_bb = self._get_img_feat(
            example['img_fname'][0], example['img_fname'][1])
        input_ids_q, type_ids_q,input_ids_qa,type_ids_qa = self._get_input_ids(example)
        outs = []
        if self.split=='val':
            qa_label = torch.tensor(example['qa_target'])
            qar_label = torch.tensor(example['qar_target'])
            input_ids_as = example['input_ids_as']
            for index, input_ids_a in enumerate(input_ids_as):
                if index == qa_label:
                    target = torch.tensor([1]).long()
                else:
                    target = torch.tensor([0]).long()

                input_ids = [self.txt_db.cls_] + copy.deepcopy(input_ids_q) + \
                            [self.txt_db.sep] + input_ids_a + [self.txt_db.sep]
                # type_id
                # 0 -- question
                # 1 -- region
                # 2 -- answer
                # 3 -- rationale

                txt_type_ids = [0] + type_ids_q + [2] * (
                        len(input_ids_a) + 2)
                attn_masks = torch.ones(
                    len(input_ids) + num_bb, dtype=torch.long)
                input_ids = torch.tensor(input_ids)
                txt_type_ids = torch.tensor(txt_type_ids)
                qa_token=torch.tensor([1]).long()

                outs.append(
                    (input_ids, txt_type_ids,
                     img_feat, img_pos_feat,
                     attn_masks, target,qa_token,qid))
            input_ids_rs = example['input_ids_rs']
            for i in range(input_ids_qa):
                for index, input_ids_r in enumerate(input_ids_rs):
                    if index == qar_label:
                        target = torch.tensor([1]).long()
                    else:
                        target = torch.tensor([0]).long()

                    input_ids = [self.txt_db.cls_] + copy.deepcopy(input_ids_qa[i]) + \
                                [self.txt_db.sep] + input_ids_r + [self.txt_db.sep]

                    txt_type_ids = [0] + type_ids_qa + [3] * (
                            len(input_ids_r[i]) + 2)
                    attn_masks = torch.ones(
                        len(input_ids) + num_bb, dtype=torch.long)
                    input_ids = torch.tensor(input_ids)
                    txt_type_ids = torch.tensor(txt_type_ids)
                    qa_token = torch.tensor([0]).long()
                    outs.append(
                        (input_ids, txt_type_ids,
                         img_feat, img_pos_feat,
                         attn_masks, target,qa_token,qid))
        else:
            input_ids_as = example['input_ids_as']
            for index, input_ids_a in enumerate(input_ids_as):
                target = torch.tensor([2]).long()
                input_ids = [self.txt_db.cls_] + copy.deepcopy(input_ids_q) + \
                            [self.txt_db.sep] + input_ids_a + [self.txt_db.sep]
                txt_type_ids = [0] + type_ids_q + [2] * (
                        len(input_ids_a) + 2)
                attn_masks = torch.ones(
                    len(input_ids) + num_bb, dtype=torch.long)
                input_ids = torch.tensor(input_ids)
                txt_type_ids = torch.tensor(txt_type_ids)
                qa_token = torch.tensor([1]).long()
                outs.append(
                    (input_ids, txt_type_ids,
                     img_feat, img_pos_feat,
                     attn_masks, target,qa_token,qid))
            input_ids_rs = example['input_ids_rs']
            for i in range(input_ids_qa):
                for index, input_ids_r in enumerate(input_ids_rs):
                    target = torch.tensor([2]).long()
                    input_ids = [self.txt_db.cls_] + copy.deepcopy(input_ids_qa[i]) + \
                                [self.txt_db.sep] + input_ids_r + [self.txt_db.sep]
                    txt_type_ids = [0] + type_ids_qa[i] + [3] * (
                            len(input_ids_r) + 2)
                    attn_masks = torch.ones(
                        len(input_ids) + num_bb, dtype=torch.long)
                    input_ids = torch.tensor(input_ids)
                    txt_type_ids = torch.tensor(txt_type_ids)
                    qa_token = torch.tensor([0]).long()
                    outs.append(
                        (input_ids, txt_type_ids,
                         img_feat, img_pos_feat,
                         attn_masks, target,qa_token,qid))
        return tuple(outs)

def Comet_qar_val_vcr_collate(inputs):
    (input_ids, txt_type_ids, img_feats,
     img_pos_feats, attn_masks, targets,qa_token,qid) = map(list, unzip(concat(inputs)))

    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    txt_type_ids = pad_sequence(
        txt_type_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)
    # image batches
    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)
    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)

    targets = torch.stack(targets, dim=0)
    qa_token = torch.stack(qa_token, dim=0)
    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'txt_type_ids': txt_type_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'targets': targets,
             'qa_token':qa_token,
             'qids': qid
             }
    batch = move_to_cuda(batch)
    return batch


