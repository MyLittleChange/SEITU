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

class RA_VcrDataset(VcrDetectFeatTxtTokDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # assert self.task != "qa,qar",\
        #     "loading training dataset with each task separately"

    def _get_input_ids(self, txt_dump):
        # text input
        input_ids_q = txt_dump['input_ids']
        type_ids_q = [0]*len(input_ids_q)
        input_ids_as = txt_dump['input_ids_as']
        input_ids_as_all = []
        # 把所有答案拼在问题后面，用sep隔开
        # 用2作为type
        # type_id
        # 0 -- question
        # 1 -- region
        # 2 -- answer
        # 3 -- rationale
        for ans_ids in input_ids_as:
            input_ids_as_all += [self.txt_db.sep] + copy.deepcopy(
                ans_ids)
        type_ids_as_all = [2] * len(input_ids_as_all)
        type_ids_q += type_ids_as_all
        input_ids_q += input_ids_as_all
        return input_ids_q, type_ids_q

    def __getitem__(self, i):
        """
        [[txt, img1],
         [txt, img2]]
        """
        qid = self.ids[i]
        example = super().__getitem__(i)
        img_feat, img_pos_feat, num_bb = self._get_img_feat(
            example['img_fname'][0], example['img_fname'][1])
        input_ids_q, type_ids_q = self._get_input_ids(
            example)
        input_ids = [self.txt_db.cls_] + copy.deepcopy(input_ids_q) +  [self.txt_db.sep]
        txt_type_ids = [0] + type_ids_q+[2]
        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)
        input_ids = torch.tensor(input_ids)
        txt_type_ids = torch.tensor(txt_type_ids)
        a_target=torch.tensor(example['qa_target'])
        # a_target=torch.tensor([0,0,0,0]).long()
        # a_target[qa_target]=1
        r_target = [self.txt_db.cls_] + example['input_ids_rs'][example['rationale_label']] + [self.txt_db.sep]
        r_len = len(r_target)
        r_target = torch.tensor(r_target)
        outs = []
        outs.append(
            (input_ids, txt_type_ids,
             img_feat, img_pos_feat,
             attn_masks, a_target,r_target,r_len,qid))

        return tuple(outs)

def RA_vcr_collate(inputs):
    (input_ids, txt_type_ids, img_feats,
     img_pos_feats, attn_masks, a_targets,r_targets,r_lens,qids) = map(list, unzip(concat(inputs)))

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
    r_targets=pad_sequence(
        r_targets, batch_first=True, padding_value=0)
    r_labels=r_targets[:,1:]
    pad = torch.zeros((r_labels.size(0),1), dtype=torch.int64)
    r_labels=torch.cat((r_labels,pad),dim=1)
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
             'r_targets': r_targets,
             'r_labels':r_labels,
             'r_length':r_lens,
             'qids':qids}
    batch = move_to_cuda(batch)
    return batch


class RA_graph_VcrDataset(VcrDetectFeatTxtTokDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # assert self.task != "qa,qar",\
        #     "loading training dataset with each task separately"

    def _get_input_ids(self, txt_dump):
        # text input
        input_ids_q = txt_dump['input_ids']
        type_ids_q = [0]*len(input_ids_q)
        input_ids_as = txt_dump['input_ids_as']
        input_ids_as_all = []
        # 把所有答案拼在问题后面，用sep隔开
        # 用2作为type
        # type_id
        # 0 -- question
        # 1 -- region
        # 2 -- answer
        # 3 -- rationale
        for ans_ids in input_ids_as:
            input_ids_as_all += [self.txt_db.sep] + copy.deepcopy(
                ans_ids)
        type_ids_as_all = [2] * len(input_ids_as_all)
        type_ids_q += type_ids_as_all
        input_ids_q += input_ids_as_all

        input_region_q = txt_dump['toked_txt_region_tokens']
        input_region_as_all = []
        input_region_as = txt_dump['toked_txt_region_tokens_a']
        for regin_as in input_region_as:
            input_region_as_all += [0] + copy.deepcopy(regin_as)
        input_region_q += input_region_as_all

        vis_edge = np.zeros((len(txt_dump['objects']), len(txt_dump['objects'])))
        q_obj = list(set(input_region_q))
        q_obj.remove(0)
        for i in range(len(q_obj)):
            for j in range(i + 1, len(q_obj)):
                if q_obj[i] == q_obj[j]:
                    continue
                vis_edge[q_obj[i] - 1][q_obj[j] - 1] = 1
                vis_edge[q_obj[j] - 1][q_obj[i] - 1] = 1
        for regin_as in input_region_as:
            q_obj = list(set(regin_as))
            q_obj.remove(0)
            for i in range(len(q_obj)):
                for j in range(i + 1, len(q_obj)):
                    if q_obj[i] == q_obj[j]:
                        continue
                    vis_edge[q_obj[i] - 1][q_obj[j] - 1] = 1
                    vis_edge[q_obj[j] - 1][q_obj[i] - 1] = 1
        return input_ids_q, type_ids_q,input_region_q,vis_edge

    def __getitem__(self, i):
        """
        [[txt, img1],
         [txt, img2]]
        """
        qid = self.ids[i]
        example = super().__getitem__(i)
        img_feat, img_pos_feat, num_bb = self._get_img_feat(
            example['img_fname'][0], example['img_fname'][1])
        input_ids_q, type_ids_q,input_region_q,vis_edge = self._get_input_ids(
            example)
        input_ids = [self.txt_db.cls_] + copy.deepcopy(input_ids_q) +  [self.txt_db.sep]
        txt_type_ids = [0] + type_ids_q+[2]
        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)
        input_ids = torch.tensor(input_ids)
        txt_type_ids = torch.tensor(txt_type_ids)
        a_target=torch.tensor(example['qa_target'])
        r_target = [self.txt_db.cls_] + example['input_ids_rs'][example['rationale_label']] + [self.txt_db.sep]
        r_len = len(r_target)
        r_target = torch.tensor(r_target)
        outs = []
        outs.append(
            (input_ids, txt_type_ids,
             img_feat, img_pos_feat,
             attn_masks, a_target,r_target,r_len,qid,input_region_q,vis_edge))

        return tuple(outs)

def RA_graph_vcr_collate(inputs):
    (input_ids, txt_type_ids, img_feats,
     img_pos_feats, attn_masks, a_targets,r_targets,r_lens,qids,input_region_q,vis_edge) = map(list, unzip(concat(inputs)))

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
    r_targets=pad_sequence(
        r_targets, batch_first=True, padding_value=0)
    r_labels=r_targets[:,1:]
    pad = torch.zeros((r_labels.size(0),1), dtype=torch.int64)
    r_labels=torch.cat((r_labels,pad),dim=1)
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
             'r_targets': r_targets,
             'r_labels':r_labels,
             'r_length':r_lens,
             'qids':qids,
             'input_region': input_region_q,
             'vis_edge': vis_edge
             }
    batch = move_to_cuda(batch)
    return batch

class RA_graph_triple_VcrDataset(VcrDetectFeatTxtTokDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # assert self.task != "qa,qar",\
        #     "loading training dataset with each task separately"

    def _get_input_ids(self, txt_dump):
        # text input
        input_ids_q = txt_dump['input_ids']
        type_ids_q = [0]*len(input_ids_q)
        input_ids_as = txt_dump['input_ids_as']
        input_ids_as_all = []
        input_ids_q_golden=input_ids_q+[self.txt_db.sep]+copy.deepcopy(txt_dump['input_ids_rs'][txt_dump['rationale_label']])
        type_ids_q_golden = type_ids_q + [3] * (len(txt_dump['input_ids_rs'][txt_dump['rationale_label']]) + 1)


        for ans_ids in input_ids_as:
            input_ids_as_all += [self.txt_db.sep] + copy.deepcopy(
                ans_ids)
        type_ids_as_all = [2] * len(input_ids_as_all)
        type_ids_q += type_ids_as_all
        type_ids_q_golden+=type_ids_as_all
        input_ids_q_golden+=input_ids_as_all
        input_ids_q += input_ids_as_all


        input_region_q = txt_dump['toked_txt_region_tokens']
        input_region_q_golden = input_region_q + [0] + copy.deepcopy(
            txt_dump['toked_txt_region_tokens_r'][txt_dump['rationale_label']])
        input_region_as_all = []
        input_region_as = txt_dump['toked_txt_region_tokens_a']
        for regin_as in input_region_as:
            input_region_as_all += [0] + copy.deepcopy(regin_as)
        input_region_q += input_region_as_all
        input_region_q_golden+=input_region_as_all

        vis_edge = np.zeros((len(txt_dump['objects']), len(txt_dump['objects'])))
        q_obj = list(set(input_region_q))
        q_obj.remove(0)
        for i in range(len(q_obj)):
            for j in range(i + 1, len(q_obj)):
                if q_obj[i] == q_obj[j]:
                    continue
                vis_edge[q_obj[i] - 1][q_obj[j] - 1] = 1
                vis_edge[q_obj[j] - 1][q_obj[i] - 1] = 1
        for regin_as in input_region_as:
            q_obj = list(set(regin_as))
            q_obj.remove(0)
            for i in range(len(q_obj)):
                for j in range(i + 1, len(q_obj)):
                    if q_obj[i] == q_obj[j]:
                        continue
                    vis_edge[q_obj[i] - 1][q_obj[j] - 1] = 1
                    vis_edge[q_obj[j] - 1][q_obj[i] - 1] = 1
        return input_ids_q, type_ids_q,input_region_q,vis_edge,input_ids_q_golden,input_region_q_golden,type_ids_q_golden

    def __getitem__(self, i):
        """
        [[txt, img1],
         [txt, img2]]
        """
        qid = self.ids[i]
        example = super().__getitem__(i)
        img_feat, img_pos_feat, num_bb = self._get_img_feat(
            example['img_fname'][0], example['img_fname'][1])
        input_ids_q, type_ids_q,input_region_q,vis_edge,input_ids_q_golden,input_region_q_golden,type_ids_q_golden= self._get_input_ids(
            example)
        input_ids = [self.txt_db.cls_] + copy.deepcopy(input_ids_q) +  [self.txt_db.sep]
        input_ids_golden= [self.txt_db.cls_] + copy.deepcopy(input_ids_q_golden) +  [self.txt_db.sep]
        txt_type_ids = [0] + type_ids_q+[2]
        input_region_q=[0]+input_region_q+[0]
        input_region_q_golden=[0]+input_region_q_golden+[0]
        txt_type_ids_golden = [0] + type_ids_q_golden + [2]
        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)
        attn_masks_golden = torch.ones(len(input_ids_golden) + num_bb, dtype=torch.long)
        input_ids = torch.tensor(input_ids)
        input_ids_golden = torch.tensor(input_ids_golden)
        txt_type_ids = torch.tensor(txt_type_ids)
        txt_type_ids_golden = torch.tensor(txt_type_ids_golden)
        a_target=torch.tensor(example['qa_target'])
        r_target = [self.txt_db.cls_] + example['input_ids_rs'][example['rationale_label']] + [self.txt_db.sep]
        r_len = len(r_target)
        r_target = torch.tensor(r_target)
        outs = []
        outs.append(
            (input_ids, txt_type_ids,
             img_feat, img_pos_feat,
             attn_masks, a_target,r_target,r_len,qid,input_region_q,vis_edge,input_ids_golden,txt_type_ids_golden,input_region_q_golden,attn_masks_golden))

        return tuple(outs)

def RA_graph_triple_vcr_collate(inputs):
    (input_ids, txt_type_ids, img_feats,
     img_pos_feats, attn_masks, a_targets,r_targets,r_lens,qids,input_region_q,vis_edge,input_ids_golden,txt_type_ids_golden,input_region_q_golden,attn_masks_golden) = map(list, unzip(concat(inputs)))

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
    r_targets=pad_sequence(
        r_targets, batch_first=True, padding_value=0)
    r_labels=r_targets[:,1:]
    pad = torch.zeros((r_labels.size(0),1), dtype=torch.int64)
    r_labels=torch.cat((r_labels,pad),dim=1)
    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    a_targets = torch.stack(a_targets,dim=0)

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    txt_lens_golden = [i.size(0) for i in input_ids_golden]
    input_ids_golden = pad_sequence(input_ids_golden, batch_first=True, padding_value=0)
    txt_type_ids_golden = pad_sequence(
        txt_type_ids_golden, batch_first=True, padding_value=0)
    position_ids_golden = torch.arange(0, input_ids_golden.size(1), dtype=torch.long
                                ).unsqueeze(0)
    attn_masks_golden = pad_sequence(attn_masks_golden, batch_first=True, padding_value=0)
    bs, max_tl = input_ids_golden.size()
    out_size = attn_masks_golden.size(1)
    gather_index_golden = get_gather_index(txt_lens_golden, num_bbs, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'txt_type_ids': txt_type_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'a_targets': a_targets,
             'r_targets': r_targets,
             'r_labels':r_labels,
             'r_length':r_lens,
             'qids':qids,
             'input_region': input_region_q,
             'vis_edge': vis_edge,
             'input_ids_golden': input_ids_golden,
             'input_region_golden': input_region_q_golden,
             'txt_type_ids_golden': txt_type_ids_golden,
             'position_ids_golden': position_ids_golden,
             'attn_masks_golden': attn_masks_golden,
             'gather_index_golden': gather_index_golden,
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

class QRA_VcrEvalDataset(VcrDetectFeatTxtTokDataset):
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
        input_ids_rs = txt_dump['input_ids_rs'][txt_dump["qar_target"]]
        type_ids_r=[3]*len(input_ids_rs)
        for index, input_ids_a in enumerate(input_ids_as):
            curr_input_ids_qa = [self.txt_db.cls_] + copy.deepcopy(input_ids_q) +\
                [self.txt_db.sep]+copy.deepcopy(input_ids_rs) +[self.txt_db.sep] + input_ids_a + [self.txt_db.sep]
            curr_type_ids_qa = [0] + type_ids_q +type_ids_r+[3]+ [2]*(
                len(input_ids_a)+2)
            input_ids_for_choices.append(curr_input_ids_qa)
            type_ids_for_choices.append(curr_type_ids_qa)
        return input_ids_for_choices, type_ids_for_choices

    def __getitem__(self, i):
        qid = self.ids[i]
        example = super().__getitem__(i)
        img_feat, img_pos_feat, num_bb = self._get_img_feat(
            example['img_fname'][0], example['img_fname'][1])

        input_ids_for_choices, type_ids_for_choices = self._get_input_ids(
            example)
        qa_target = torch.tensor([int(example["qa_target"])])

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

        return tuple(outs), qid, qa_target

def QRA_vcr_eval_collate(inputs):
    (input_ids, txt_type_ids, img_feats,
     img_pos_feats, attn_masks) = map(
        list, unzip(concat(outs for outs, _, _ in inputs)))

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
        [t for _, _, t in inputs], dim=0)
    qids = [id_ for _, id_, _ in inputs]

    return {'qids': qids,
            'input_ids': input_ids,
            'txt_type_ids': txt_type_ids,
            'position_ids': position_ids,
            'img_feat': img_feat,
            'img_pos_feat': img_pos_feat,
            'attn_masks': attn_masks,
            'gather_index': gather_index,
            'qa_targets': qa_targets}


class QRA_gen_VcrEvalDataset(VcrDetectFeatTxtTokDataset):
    def __init__(self, split,gen_file, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.split = split
        with open(gen_file,'r') as f:
            data=f.readlines()
        self.gen_dict = json.loads(data[0])
        self.w2i,self.i2w=make_dict()


    def _get_input_ids(self, txt_dump,qid):
        # text input
        input_ids_for_choices = []
        type_ids_for_choices = []
        input_ids_q = txt_dump['input_ids']
        type_ids_q = [0]*len(input_ids_q)
        input_ids_as = txt_dump['input_ids_as']

        #input_ids_rs = txt_dump['input_ids_rs'][txt_dump["qar_target"]]
        input_token_rs=self.gen_dict[qid]['pred']
        input_ids_rs=[self.w2i[item] for item in input_token_rs.split(' ')[:-1]]
        type_ids_r=[3]*len(input_ids_rs)
        for index, input_ids_a in enumerate(input_ids_as):
            curr_input_ids_qa = [self.txt_db.cls_] + copy.deepcopy(input_ids_q) +\
                [self.txt_db.sep]+copy.deepcopy(input_ids_rs) +[self.txt_db.sep] + input_ids_a + [self.txt_db.sep]
            curr_type_ids_qa = [0] + type_ids_q +type_ids_r+[3]+ [2]*(
                len(input_ids_a)+2)
            input_ids_for_choices.append(curr_input_ids_qa)
            type_ids_for_choices.append(curr_type_ids_qa)
        return input_ids_for_choices, type_ids_for_choices

    def __getitem__(self, i):
        qid = self.ids[i]
        example = super().__getitem__(i)
        img_feat, img_pos_feat, num_bb = self._get_img_feat(
            example['img_fname'][0], example['img_fname'][1])

        input_ids_for_choices, type_ids_for_choices = self._get_input_ids(
            example,qid)
        qa_target = torch.tensor([int(example["qa_target"])])

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

        return tuple(outs), qid, qa_target

def QRA_gen_vcr_eval_collate(inputs):
    (input_ids, txt_type_ids, img_feats,
     img_pos_feats, attn_masks) = map(
        list, unzip(concat(outs for outs, _, _ in inputs)))

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
        [t for _, _, t in inputs], dim=0)
    qids = [id_ for _, id_, _ in inputs]

    return {'qids': qids,
            'input_ids': input_ids,
            'txt_type_ids': txt_type_ids,
            'position_ids': position_ids,
            'img_feat': img_feat,
            'img_pos_feat': img_pos_feat,
            'attn_masks': attn_masks,
            'gather_index': gather_index,
            'qa_targets': qa_targets}



