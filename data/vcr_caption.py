"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

VCR dataset
"""
import copy
import json
import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip
from cytoolz import concat
from .data import (DetectFeatTxtTokDataset, TxtTokLmdb, DetectFeatLmdb,
                   TxtLmdb, get_ids_and_lens, pad_tensors,
                   get_gather_index)
from utils.make_dict import make_dict
import numpy as np
def dict_slice(adict):
    keys = adict.keys()
    dict_slice = {}
    for k in keys:
        dict_slice[k] = adict[k]
        if len(dict_slice)==100:
            break
    return dict_slice

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


class VcrDataset(VcrDetectFeatTxtTokDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # assert self.task != "qa,qar",\
        #     "loading training dataset with each task separately"

    def _get_input_ids(self, txt_dump):
        # text input
        input_ids_q = txt_dump['input_ids']
        type_ids_q = [0]*len(input_ids_q)
        input_ids_as = txt_dump['input_ids_as']
        if self.task == "qar":
            input_ids_rs = txt_dump['input_ids_rs']
            answer_label = txt_dump['qa_target']
            assert answer_label >= 0, "answer_label < 0"
            input_ids_gt_a = [self.txt_db.sep] + copy.deepcopy(
                input_ids_as[answer_label])
            type_ids_gt_a = [2] * len(input_ids_gt_a)
            type_ids_q += type_ids_gt_a
            input_ids_q += input_ids_gt_a
            input_ids_for_choices = input_ids_rs
        else:
            input_ids_for_choices = input_ids_as
        return input_ids_q, input_ids_for_choices, type_ids_q

    def __getitem__(self, i):
        """
        [[txt, img1],
         [txt, img2]]
        """
        example = super().__getitem__(i)
        img_feat, img_pos_feat, num_bb = self._get_img_feat(
            example['img_fname'][0], example['img_fname'][1])
        input_ids_q, input_ids_for_choices, type_ids_q = self._get_input_ids(
            example)
        label = example['%s_target' % (self.task)]

        outs = []
        for index, input_ids_a in enumerate(input_ids_for_choices):
            if index == label:
                target = torch.tensor([1]).long()
            else:
                target = torch.tensor([0]).long()
            input_ids = [self.txt_db.cls_] + copy.deepcopy(input_ids_q) +\
                [self.txt_db.sep] + input_ids_a + [self.txt_db.sep]
            # type_id
            # 0 -- question
            # 1 -- region
            # 2 -- answer
            # 3 -- rationale
            type_id_for_choice = 3 if type_ids_q[-1] == 2 else 2
            txt_type_ids = [0] + type_ids_q + [type_id_for_choice]*(
                len(input_ids_a)+2)
            attn_masks = torch.ones(
                len(input_ids) + num_bb, dtype=torch.long)
            input_ids = torch.tensor(input_ids)
            txt_type_ids = torch.tensor(txt_type_ids)

            outs.append(
                (input_ids, txt_type_ids,
                 img_feat, img_pos_feat,
                 attn_masks, target))

        return tuple(outs)

def vcr_collate(inputs):
    (input_ids, txt_type_ids, img_feats,
     img_pos_feats, attn_masks, targets) = map(list, unzip(concat(inputs)))

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
             'targets': targets}
    batch = move_to_cuda(batch)
    return batch

class Caption_VcrDataset(VcrDetectFeatTxtTokDataset):
    def __init__(self, tokenizer,*args, **kwargs):
        super().__init__(*args, **kwargs)
        # assert self.task != "qa,qar",\
        #     "loading training dataset with each task separately"
        caption_path='/raid/yq/UNITER/pretrain/caption_result_train_3_filter.json'
        with open(caption_path,'r') as f:
            self.caption=json.load(f)
        self.tokenizer=tokenizer
        special_token_path= '/raid/yq/UNITER/pretrain/txt_db/vcr_train.db/special_tokens.json'
        with open(special_token_path,'r') as f:
            self.num_special_tokens=f.readline()
        self.max_cap=100

    def bert_tokenize(self, text,region):
        ids = []
        region_index = np.nonzero(region)[0]
        redion_ids=[region[i]-1 for i in region_index]
        for sent in text:
            region_in=False
            #判断是否存在对应实体
            for id in redion_ids:
                if str(id) in sent:
                    region_in=True
                    break
            if region_in==False:
                continue
            word_list=sent.strip().split()
            for i in range(len(word_list)):
                if 'person_' in word_list[i]:
                        ws_id=28996+int(word_list[i][7])
                        ids.extend([ws_id])
                else:
                    ws = self.tokenizer.tokenize(word_list[i])
                    if not ws:
                        # some special char
                        continue
                    ids.extend(self.tokenizer.convert_tokens_to_ids(ws))
            ids.extend(self.tokenizer.convert_tokens_to_ids(['.']))
        return ids[:-1]

    def _get_input_ids(self, txt_dump):
        # text input
        input_ids_q = txt_dump['input_ids']
        type_ids_q = [0]*len(input_ids_q)

        #增加caption作为解释
        try:
            caption=self.caption[txt_dump['img_fname'][1][:-4]+'.npz'][0]['caption']
        except:
            caption=[]
        caption_id=self.bert_tokenize(caption,txt_dump['toked_txt_region_tokens'])
        caption_id=caption_id[:self.max_cap]
        # caption_id=[119 for i in range(100)]
        input_ids_q=input_ids_q+[self.txt_db.sep]+copy.deepcopy(caption_id)
        type_ids_q = type_ids_q + [3] * (len(caption_id) + 1)

        return input_ids_q, type_ids_q

    def __getitem__(self, i):
        """
        [[txt, img1],
         [txt, img2]]
        """

        example = super().__getitem__(i)
        img_feat, img_pos_feat, num_bb = self._get_img_feat(
            example['img_fname'][0], example['img_fname'][1])
        input_ids_q, type_ids_q= self._get_input_ids(example)
        label=torch.tensor(example['qa_target'])

        # 二分类
        outs = []
        input_ids_as = example['input_ids_as']
        match_iter=torch.tensor(example['answer_match_iter'])

        for index, input_ids_a in enumerate(input_ids_as):
            if index == label:
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

            outs.append(
                (input_ids, txt_type_ids,
                 img_feat, img_pos_feat,
                 attn_masks, target,match_iter))


        return tuple(outs)

def Caption_vcr_collate(inputs):
    (input_ids, txt_type_ids, img_feats,
     img_pos_feats, attn_masks, a_targets,match_iter) = map(list, unzip(concat(inputs)))

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
    a_targets = torch.stack(a_targets,dim=0)
    match_iter = torch.stack(match_iter, dim=0)

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
             'a_targets': a_targets,
             'match_iter':match_iter
             }
    batch = move_to_cuda(batch)
    return batch

class Caption_val_VcrDataset(VcrDetectFeatTxtTokDataset):
    def __init__(self, tokenizer,*args, **kwargs):
        super().__init__(*args, **kwargs)
        # assert self.task != "qa,qar",\
        #     "loading training dataset with each task separately"
        caption_path='/raid/yq/UNITER/pretrain/caption_result_val_10_filter.json'
        with open(caption_path,'r') as f:
            self.caption=json.load(f)
        self.tokenizer=tokenizer
        special_token_path= '/raid/yq/UNITER/pretrain/txt_db/vcr_train.db/special_tokens.json'
        with open(special_token_path,'r') as f:
            self.num_special_tokens=f.readline()
        self.max_cap=100

    def bert_tokenize(self, text,region):
        ids = []
        region_index = np.nonzero(region)[0]
        redion_ids=[region[i]-1 for i in region_index]
        for sent in text:
            region_in=False
            #判断是否存在对应实体
            for id in redion_ids:
                if str(id) in sent:
                    region_in=True
                    break
            if region_in==False:
                continue
            word_list=sent.strip().split()
            for i in range(len(word_list)):
                if 'person_' in word_list[i]:
                        ws_id=28996+int(word_list[i][7])
                        ids.extend([ws_id])
                else:
                    ws = self.tokenizer.tokenize(word_list[i])
                    if not ws:
                        # some special char
                        continue
                    ids.extend(self.tokenizer.convert_tokens_to_ids(ws))
            ids.extend(self.tokenizer.convert_tokens_to_ids(['.']))
        return ids[:-1]

    def _get_input_ids(self, txt_dump):
        # text input
        input_ids_q = txt_dump['input_ids']
        type_ids_q = [0]*len(input_ids_q)

        #增加caption作为解释
        try:
            caption=self.caption[txt_dump['img_fname'][1][:-4]+'.npz'][0]['caption']
        except:
            caption=[]
        caption_id=self.bert_tokenize(caption,txt_dump['toked_txt_region_tokens'])
        caption_id=caption_id[:self.max_cap]
        # caption_id=[119 for i in range(100)]
        input_ids_q=input_ids_q+[self.txt_db.sep]+copy.deepcopy(caption_id)
        type_ids_q = type_ids_q + [3] * (len(caption_id) + 1)


        return input_ids_q, type_ids_q

    def __getitem__(self, i):
        """
        [[txt, img1],
         [txt, img2]]
        """

        example = super().__getitem__(i)
        img_feat, img_pos_feat, num_bb = self._get_img_feat(
            example['img_fname'][0], example['img_fname'][1])
        input_ids_q, type_ids_q= self._get_input_ids(example)
        label=torch.tensor(example['qa_target'])



        # 二分类
        outs = []
        input_ids_as = example['input_ids_as']
        for index, input_ids_a in enumerate(input_ids_as):
            if index == label:
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

            outs.append(
                (input_ids, txt_type_ids,
                 img_feat, img_pos_feat,
                 attn_masks, target))


        return tuple(outs)

def Caption_val_vcr_collate(inputs):
    (input_ids, txt_type_ids, img_feats,
     img_pos_feats, attn_masks, a_targets) = map(list, unzip(concat(inputs)))

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
    a_targets = torch.stack(a_targets,dim=0)

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
             'a_targets': a_targets,
             }
    batch = move_to_cuda(batch)
    return batch


class Caption_four_VcrDataset(VcrDetectFeatTxtTokDataset):
    def __init__(self, tokenizer,*args, **kwargs):
        super().__init__(*args, **kwargs)
        # assert self.task != "qa,qar",\
        #     "loading training dataset with each task separately"
        caption_path='/raid/yq/UNITER/pretrain/caption_result_train_3_filter.json'
        with open(caption_path,'r') as f:
            self.caption=json.load(f)
        self.tokenizer=tokenizer
        special_token_path= '/raid/yq/UNITER/pretrain/txt_db/vcr_train.db/special_tokens.json'
        with open(special_token_path,'r') as f:
            self.num_special_tokens=f.readline()
        self.max_cap=100

    def bert_tokenize(self, text,region):
        ids = []
        region_index = np.nonzero(region)[0]
        redion_ids=[region[i]-1 for i in region_index]
        for sent in text:
            region_in=False
            #判断是否存在对应实体
            for id in redion_ids:
                if str(id) in sent:
                    region_in=True
                    break
            if region_in==False:
                continue
            word_list=sent.strip().split()
            for i in range(len(word_list)):
                if 'person_' in word_list[i]:
                        ws_id=28996+int(word_list[i][7])
                        ids.extend([ws_id])
                else:
                    ws = self.tokenizer.tokenize(word_list[i])
                    if not ws:
                        # some special char
                        continue
                    ids.extend(self.tokenizer.convert_tokens_to_ids(ws))
            ids.extend(self.tokenizer.convert_tokens_to_ids(['.']))
        return ids[:-1]

    def _get_input_ids(self, txt_dump):
        # text input
        input_ids_q = txt_dump['input_ids']
        type_ids_q = [0]*len(input_ids_q)

        # #增加caption作为解释
        # try:
        #     caption=self.caption[txt_dump['img_fname'][1][:-4]+'.npz'][0]['caption']
        # except:
        #     caption=[]
        # caption_id=self.bert_tokenize(caption,txt_dump['toked_txt_region_tokens'])
        # caption_id=caption_id[:self.max_cap]
        # # caption_id=[119 for i in range(100)]
        # input_ids_q=input_ids_q+[self.txt_db.sep]+copy.deepcopy(caption_id)
        # type_ids_q = type_ids_q + [3] * (len(caption_id) + 1)

        input_ids_as = txt_dump['input_ids_as']
        input_ids_as_all = []
        # type_id
        # 0 -- question
        # 1 -- region
        # 2 -- answer
        # 3 -- rationale
        tmp_len=len(input_ids_q)
        ans_sep =[tmp_len]
        for i in range(len(input_ids_as)):
            input_ids_as_all += [self.txt_db.sep] + copy.deepcopy(
                input_ids_as[i])
            tmp_len += len(input_ids_as[i]) + 1
            ans_sep.append(tmp_len)

        type_ids_as_all = [2] * len(input_ids_as_all)

        type_ids_q += type_ids_as_all
        input_ids_q += input_ids_as_all
        ans_index = [0] * len(input_ids_q)
        ans_mask = [0] * len(input_ids_q)
        for i in range(len(ans_sep)-1):
            ans_index[ans_sep[i]]=1
            ans_mask[ans_sep[i]+1:ans_sep[i+1]]=[i+1]*(ans_sep[i+1]-ans_sep[i]-1)
        return input_ids_q, type_ids_q,ans_index,ans_mask

    def __getitem__(self, i):
        """
        [[txt, img1],
         [txt, img2]]
        """

        example = super().__getitem__(i)
        img_feat, img_pos_feat, num_bb = self._get_img_feat(
            example['img_fname'][0], example['img_fname'][1])
        input_ids_q, type_ids_q,ans_index,ans_mask= self._get_input_ids(example)
        label=torch.tensor(example['qa_target'])
        target_two=torch.zeros(4,dtype=torch.int64).cuda()
        target_two[label]=1
        input_ids = [self.txt_db.cls_] + copy.deepcopy(input_ids_q) +  [self.txt_db.sep]

        txt_type_ids = [0] + type_ids_q+[2]
        ans_index=[0]+ans_index+[1]
        ans_mask=[0]+ans_mask+[0]
        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)

        input_ids = torch.tensor(input_ids)

        txt_type_ids = torch.tensor(txt_type_ids)
        ans_index=torch.tensor(ans_index)
        ans_mask = torch.tensor(ans_mask)
        match_iter = torch.tensor(example['answer_match_iter'])
        outs = []
        outs.append(
            (input_ids, txt_type_ids,
             img_feat, img_pos_feat,
             attn_masks, target_two,label,match_iter,ans_index,ans_mask))

        return tuple(outs)

def Caption_four_vcr_collate(inputs):
    (input_ids, txt_type_ids, img_feats,
     img_pos_feats, attn_masks, a_targets,a_label,match_iter,ans_index,ans_mask) = map(list, unzip(concat(inputs)))

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
    a_targets = torch.stack(a_targets,dim=0)
    a_label = torch.stack(a_label, dim=0)
    match_iter = torch.stack(match_iter, dim=0)
    ans_index = pad_sequence(ans_index, batch_first=True, padding_value=0)
    ans_mask = pad_sequence(ans_mask, batch_first=True, padding_value=0)
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
             'a_targets': a_targets,
             'match_iter':match_iter,
             'ans_index':ans_index,
             'a_label':a_label,
             'ans_mask':ans_mask
             }
    batch = move_to_cuda(batch)
    return batch


class Caption_four_val_VcrDataset(VcrDetectFeatTxtTokDataset):
    def __init__(self, tokenizer,*args, **kwargs):
        super().__init__(*args, **kwargs)
        # assert self.task != "qa,qar",\
        #     "loading training dataset with each task separately"
        caption_path='/raid/yq/UNITER/pretrain/caption_result_val_10_filter.json'
        with open(caption_path,'r') as f:
            self.caption=json.load(f)
        self.tokenizer=tokenizer
        special_token_path= '/raid/yq/UNITER/pretrain/txt_db/vcr_train.db/special_tokens.json'
        with open(special_token_path,'r') as f:
            self.num_special_tokens=f.readline()
        self.max_cap=100

    def bert_tokenize(self, text,region):
        ids = []
        region_index = np.nonzero(region)[0]
        redion_ids=[region[i]-1 for i in region_index]
        for sent in text:
            region_in=False
            #判断是否存在对应实体
            for id in redion_ids:
                if str(id) in sent:
                    region_in=True
                    break
            if region_in==False:
                continue
            word_list=sent.strip().split()
            for i in range(len(word_list)):
                if 'person_' in word_list[i]:
                        ws_id=28996+int(word_list[i][7])
                        ids.extend([ws_id])
                else:
                    ws = self.tokenizer.tokenize(word_list[i])
                    if not ws:
                        # some special char
                        continue
                    ids.extend(self.tokenizer.convert_tokens_to_ids(ws))
            ids.extend(self.tokenizer.convert_tokens_to_ids(['.']))
        return ids[:-1]

    def _get_input_ids(self, txt_dump):
        # text input
        input_ids_q = txt_dump['input_ids']
        type_ids_q = [0]*len(input_ids_q)

        # #增加caption作为解释
        # try:
        #     caption=self.caption[txt_dump['img_fname'][1][:-4]+'.npz'][0]['caption']
        # except:
        #     caption=[]
        # caption_id=self.bert_tokenize(caption,txt_dump['toked_txt_region_tokens'])
        # caption_id=caption_id[:self.max_cap]
        # # caption_id=[119 for i in range(100)]
        # input_ids_q=input_ids_q+[self.txt_db.sep]+copy.deepcopy(caption_id)
        # type_ids_q = type_ids_q + [3] * (len(caption_id) + 1)

        input_ids_as = txt_dump['input_ids_as']
        input_ids_as_all = []
        # type_id
        # 0 -- question
        # 1 -- region
        # 2 -- answer
        # 3 -- rationale
        tmp_len=len(input_ids_q)
        ans_sep =[tmp_len]
        for i in range(len(input_ids_as)):
            input_ids_as_all += [self.txt_db.sep] + copy.deepcopy(
                input_ids_as[i])
            tmp_len += len(input_ids_as[i]) + 1
            ans_sep.append(tmp_len)

        type_ids_as_all = [2] * len(input_ids_as_all)

        type_ids_q += type_ids_as_all
        input_ids_q += input_ids_as_all
        ans_index = [0] * len(input_ids_q)
        ans_mask = [0] * len(input_ids_q)
        for i in range(len(ans_sep)-1):
            ans_index[ans_sep[i]]=1
            ans_mask[ans_sep[i]+1:ans_sep[i+1]]=[i+1]*(ans_sep[i+1]-ans_sep[i]-1)
        return input_ids_q, type_ids_q,ans_index,ans_mask

    def __getitem__(self, i):
        """
        [[txt, img1],
         [txt, img2]]
        """

        example = super().__getitem__(i)
        img_feat, img_pos_feat, num_bb = self._get_img_feat(
            example['img_fname'][0], example['img_fname'][1])
        input_ids_q, type_ids_q,ans_index,ans_mask= self._get_input_ids(example)
        label=torch.tensor(example['qa_target'])
        target_two=torch.zeros(4,dtype=torch.int64).cuda()
        target_two[label]=1
        input_ids = [self.txt_db.cls_] + copy.deepcopy(input_ids_q) +  [self.txt_db.sep]

        txt_type_ids = [0] + type_ids_q+[2]
        ans_index=[0]+ans_index+[1]
        ans_mask=[0]+ans_mask+[0]
        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)

        input_ids = torch.tensor(input_ids)

        txt_type_ids = torch.tensor(txt_type_ids)
        ans_index=torch.tensor(ans_index)
        ans_mask = torch.tensor(ans_mask)
        match_iter = torch.tensor(example['answer_match_iter'])
        outs = []
        outs.append(
            (input_ids, txt_type_ids,
             img_feat, img_pos_feat,
             attn_masks, target_two,label,match_iter,ans_index,ans_mask))

        return tuple(outs)

def Caption_four_val_vcr_collate(inputs):
    (input_ids, txt_type_ids, img_feats,
     img_pos_feats, attn_masks, a_targets,a_label,match_iter,ans_index,ans_mask) = map(list, unzip(concat(inputs)))

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
    a_targets = torch.stack(a_targets,dim=0)
    a_label = torch.stack(a_label, dim=0)
    match_iter = torch.stack(match_iter, dim=0)
    ans_index = pad_sequence(ans_index, batch_first=True, padding_value=0)
    ans_mask = pad_sequence(ans_mask, batch_first=True, padding_value=0)
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
             'a_targets': a_targets,
             'match_iter':match_iter,
             'ans_index':ans_index,
             'a_label':a_label,
             'ans_mask':ans_mask
             }
    batch = move_to_cuda(batch)
    return batch


class VcrEvalDataset(VcrDetectFeatTxtTokDataset):
    def __init__(self, split, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.split = split
        # assert self.task == "qa,qar",\
        #     "loading evaluation dataset with two tasks together"

    def _get_input_ids(self, txt_dump):
        # text input
        input_ids_for_choices = []
        type_ids_for_choices = []
        input_ids_q = txt_dump['input_ids']
        type_ids_q = [0]*len(input_ids_q)
        input_ids_as = txt_dump['input_ids_as']
        input_ids_rs = txt_dump['input_ids_rs']
        for index, input_ids_a in enumerate(input_ids_as):
            curr_input_ids_qa = [self.txt_db.cls_] + copy.deepcopy(input_ids_q) +\
                [self.txt_db.sep] + input_ids_a + [self.txt_db.sep]
            curr_type_ids_qa = [0] + type_ids_q + [2]*(
                len(input_ids_a)+2)
            input_ids_for_choices.append(curr_input_ids_qa)
            type_ids_for_choices.append(curr_type_ids_qa)
        for index, input_ids_a in enumerate(input_ids_as):
            #这一段是拼解释
            curr_input_ids_qa = [self.txt_db.cls_] + copy.deepcopy(input_ids_q) +\
                [self.txt_db.sep] + input_ids_a + [self.txt_db.sep]
            curr_type_ids_qa = [0] + type_ids_q + [2]*(
                len(input_ids_a)+1)
            if (self.split == "val" and index == txt_dump["qa_target"]) or\
                    self.split == "test":
                for input_ids_r in input_ids_rs:
                    curr_input_ids_qar = copy.deepcopy(curr_input_ids_qa) +\
                        input_ids_r + [self.txt_db.sep]
                    curr_type_ids_qar = copy.deepcopy(curr_type_ids_qa) +\
                        [3]*(len(input_ids_r)+2)
                    input_ids_for_choices.append(curr_input_ids_qar)
                    type_ids_for_choices.append(curr_type_ids_qar)
        return input_ids_for_choices, type_ids_for_choices

    def __getitem__(self, i):
        qid = self.ids[i]
        example = super().__getitem__(i)
        img_feat, img_pos_feat, num_bb = self._get_img_feat(
            example['img_fname'][0], example['img_fname'][1])

        input_ids_for_choices, type_ids_for_choices = self._get_input_ids(
            example)
        qa_target = torch.tensor([int(example["qa_target"])])
        qar_target = torch.tensor([int(example["qar_target"])])

        outs = []
        for index, input_ids in enumerate(input_ids_for_choices):
            attn_masks = torch.ones(
                len(input_ids) + num_bb, dtype=torch.long)

            input_ids = torch.tensor(input_ids)
            txt_type_ids = torch.tensor(
                type_ids_for_choices[index])

            outs.append(
                (input_ids, txt_type_ids,
                 img_feat, img_pos_feat,
                 attn_masks))

        return tuple(outs), qid, qa_target, qar_target

def vcr_eval_collate(inputs):
    (input_ids, txt_type_ids, img_feats,
     img_pos_feats, attn_masks) = map(
        list, unzip(concat(outs for outs, _, _, _ in inputs)))

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

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    qa_targets = torch.stack(
        [t for _, _, t, _ in inputs], dim=0)
    qar_targets = torch.stack(
        [t for _, _, _, t in inputs], dim=0)
    qids = [id_ for _, id_, _, _ in inputs]

    return {'qids': qids,
            'input_ids': input_ids,
            'txt_type_ids': txt_type_ids,
            'position_ids': position_ids,
            'img_feat': img_feat,
            'img_pos_feat': img_pos_feat,
            'attn_masks': attn_masks,
            'gather_index': gather_index,
            'qa_targets': qa_targets,
            'qar_targets': qar_targets}




