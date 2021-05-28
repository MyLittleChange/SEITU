from .vcr import VcrDetectFeatTxtTokDataset
import torch
import copy
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from .data import TxtLmdb
import json
from toolz.sandbox import unzip
from cytoolz import concat
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
class pretrain_gpt_VcrDataset(Dataset):
    def __init__(self,db_dir):
        self.db=TxtLmdb(db_dir,readonly=True)
        # assert self.task != "qa,qar",\
        #     "loading training dataset with each task separately"

        meta = json.load(open(f'{db_dir}/meta.json', 'r'))
        self.cls_ = meta['CLS']
        self.sep = meta['SEP']
        self.mask = meta['MASK']
        self.v_range = meta['v_range']
        self.id2len=json.load( open(f'{db_dir}/id2len_qa.json') )
        self.data = self.get_db()
        self.len=len(self.data)

    def __len__(self):
        return self.len
    def __getitem__(self, i):
        """
        [[txt, img1],
         [txt, img2]]
        """
        index = i % self.len
        trg=self.data[index]
        trg= [self.cls_] + copy.deepcopy(trg) +  [self.sep]
        trg_len=len(trg)
        trg=torch.tensor(trg)
        outs = []
        outs.append(
            (trg,trg_len))
        return tuple(outs)

    def get_db(self):
        data=[]
        for qid in self.id2len.keys():
            example=self.db.__getitem__(qid)
            data.append(example['input_ids'])
            for item in example['input_ids_as']:
                data.append(item)
            for item in example['input_ids_rs']:
                data.append(item)
        return data



def pretrain_gpt_collate(inputs):
    (targets,trg_len) = map(list, unzip(concat(inputs)))

    targets=pad_sequence(
        targets, batch_first=True, padding_value=0)
    labels=targets[:,1:]
    pad = torch.zeros((labels.size(0),1), dtype=torch.int64)
    labels=torch.cat((labels,pad),dim=1)

    batch = {'targets': targets,
             'trg_length':trg_len,
             'labels':labels,}
    batch = move_to_cuda(batch)
    return batch