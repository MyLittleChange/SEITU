"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

UNITER finetuning for VCR
"""
import argparse
import json
import os
from os.path import exists, join
from time import time

import torch
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from torch.utils.data import DataLoader

from transformers.models.bert import BertTokenizer
from data import ( DetectFeatLmdb,
                  VcrTxtTokLmdb, ImageLmdbGroup )
from model.vcr_caption import UniterForVisualCommonsenseReasoning_inf
from data.vcr_comet_inf import Comet_qar_val_VcrDataset,Comet_qar_val_vcr_collate

from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from utils.misc import  parse_with_config, set_dropout
from utils.const import  IMG_DIM
NUM_SPECIAL_TOKENS = 81
from progressbar import ProgressBar
import csv

def build_dataloader(dataset, collate_fn, is_train, opts):
    batch_size = (opts.train_batch_size if is_train
                  else opts.val_batch_size)
    if is_train:
        dataloader=DataLoader(dataset,drop_last=True,batch_size=opts.num_sample_batch,num_workers=0,shuffle=True,pin_memory=opts.pin_mem,collate_fn=collate_fn)
    else:
        dataloader = DataLoader(dataset, batch_size=opts.num_sample_batch,
                                num_workers=opts.n_workers, shuffle=False,
                                pin_memory=opts.pin_mem, collate_fn=collate_fn)
    return dataloader


def load_img_feat(db_list, all_img_dbs, opts):
    db_ = db_list.split(";")
    assert len(db_) <= 2, "More than two img_dbs found"
    gt_db_path, db_path = "", ""
    for d in db_:
        if "gt" in d:
            gt_db_path = d
        else:
            db_path = d
    if gt_db_path != "":
        img_db_gt = DetectFeatLmdb(
            gt_db_path, -1, opts.max_bb, opts.min_bb, 100,
            opts.compressed_db)
        all_img_dbs.path2imgdb[gt_db_path] = img_db_gt
    else:
        img_db_gt = None
    img_db = all_img_dbs[db_path] if db_path != "" else None
    all_img_dbs.path2imgdb[db_path] = img_db
    return img_db, img_db_gt


@torch.no_grad()
def validate(model_qa, model_qar,val_loader,args):
    val_pbar=ProgressBar(n_total=len(val_loader),desc='eval')
    LOGGER.info("start running validation...")
    total_ex=0
    n_correct_qa=0
    n_correct_qar_golden = 0
    n_correct_qar_join = 0
    for i, batch in enumerate(val_loader):
        total_ex += args.num_sample_batch
        with torch.no_grad():
            rank_scores_qa= model_qa(batch,'qa')
            rank_scores_qar = model_qar(batch, 'qar')
        qa_token = batch['qa_token']
        qa_mask = qa_token == 1
        qar_mask = qa_token == 0
        target=batch['target']
        qa_target=torch.masked_select(target,qa_mask)
        qar_target=torch.masked_select(target,qar_mask)

        rank_scores_qa=rank_scores_qa[:,1:]
        out=rank_scores_qa.view(rank_scores_qa.shape[0]//4,-1)
        tar=qa_target.view(qa_target.shape[0]//4,-1)
        qa_output=out.max(dim=-1)[1]
        qa_ans=np.nonzero(tar)[:,1]
        #个数为bs
        matched_qa = qa_output == qa_ans
        n_correct_qa += matched_qa.sum().item()

        # qar的正确率有两种，一个是输入为golden的正确率，另一个则是两个都正确的概率
        rank_scores_qar = rank_scores_qar[:, 1:]
        out = rank_scores_qar.view(rank_scores_qa.shape[0] // 4, -1)
        qar_output = out.max(dim=-1)[1]
        qar_tar = qar_target.view(qar_target.shape[0] // 4, -1)
        qar_ans = np.nonzero(qar_tar)[:, 1][::4]
        #转成了bs
        #output个数为bs*4
        for i in range(qa_ans):
            qa_inf=qa_ans[i]
            if qar_output[i*4+qa_inf]==qar_ans[i]:
                #i*4表示题号，加表示a对应的选项
                n_correct_qar_golden+=1
                if qa_inf==qar_output[i]:
                    n_correct_qar_join+=1
        val_pbar(step=i, info={'qa_acc':n_correct_qa/total_ex,'qar_golden_acc':n_correct_qar_golden/total_ex,'qa_join_acc':n_correct_qar_join/total_ex})
    return n_correct_qa

@torch.no_grad()
def predict(model_qa, model_qar,val_loader,args):
    val_pbar=ProgressBar(n_total=len(val_loader),desc='eval')
    LOGGER.info("start running validation...")
    total_ex=0
    soft=torch.nn.Softmax(dim=-1)
    path='result_test.csv'
    with open(path, 'a+') as f:
        csv_write = csv.writer(f)
        data_roe=["annot_id","answer_0","answer_1","answer_2","answer_3","rationale_conditioned_on_a0_0","rationale_conditioned_on_a0_1","rationale_conditioned_on_a0_2","rationale_conditioned_on_a0_3","rationale_conditioned_on_a1_0","rationale_conditioned_on_a1_1","rationale_conditioned_on_a1_2","rationale_conditioned_on_a1_3","rationale_conditioned_on_a2_0","rationale_conditioned_on_a2_1","rationale_conditioned_on_a2_2","rationale_conditioned_on_a2_3","rationale_conditioned_on_a3_0","rationale_conditioned_on_a3_1","rationale_conditioned_on_a3_2","rationale_conditioned_on_a3_3"]

    for i, batch in enumerate(val_loader):
        total_ex += args.num_sample_batch
        with torch.no_grad():
            rank_scores_qa= model_qa(batch,'qa')
            rank_scores_qar = model_qar(batch, 'qar')
        qids=batch['qids']
        rank_scores_qa=rank_scores_qa[:,1:]
        out=rank_scores_qa.view(rank_scores_qa.shape[0]//4,-1)
        ans_out=soft(out)

        # qar的正确率有两种，一个是输入为golden的正确率，另一个则是两个都正确的概率
        rank_scores_qar = rank_scores_qar[:, 1:]
        out = rank_scores_qar.view(rank_scores_qa.shape[0] // 4, -1)
        rs_out=soft(out)

        with open(path,'a+') as f:
            csv_write=csv.writer(f)
            for i in range(ans_out.shape[0]):
                qid=qid[i*20]
                tmp_ans=ans_out[i]
                tmp_rs=rs_out[i*4:i*4+4]



        val_pbar(step=i)
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--GPT_type", default=1, type=int)
    parser.add_argument('--num_gpu', default=1, type=int)
    parser.add_argument('--compressed_db', action='store_true',
                        help='use compressed LMDB')
    parser.add_argument('--do_predict', action='store_true',
                        help='进行测试')
    parser.add_argument("--encoder_config",
                        default='./config/train-vcr-large-4gpu.json', type=str,
                        help="json file for model architecture")
    parser.add_argument("--model_qa_checkpoint",
                        default="/raid/yq/UNITER/outputs/qa_comet/ckpt/UNITER-GPT-checkpoint-140000-.tar", type=str,
                        help="qa model")
    parser.add_argument("--model_qar_checkpoint",
                        default="/raid/yq/UNITER/outputs/qar_comet/ckpt/UNITER-checkpoint-120000-.tar", type=str,
                        help="qar model")

    parser.add_argument("--optimizer_checkpoint",
                        default=None, type=str,
                        help="pretrained model")

    parser.add_argument("--task",
                        default='qa', type=str,
                        choices=['qa', 'qar','qra','qr'],
                        help="which setting is checkpoint from")

    parser.add_argument(
        "--output_dir", default='./outputs', type=str,
        help="The output directory where the model checkpoints will be "
             "written.")
    parser.add_argument(
        "--global_step", default=0, type=int,
        help="")
    parser.add_argument(
        "--val_txt_db", default='./pretrain/txt_db/vcr_val.db', type=str,
        help="The output directory where the model checkpoints will be "
             "written.")
    parser.add_argument(
        "--val_img_db", default='./pretrain/img_db/vcr_val;./pretrain/img_db/vcr_gt_val', type=str,
        help="The output directory where the model checkpoints will be "
             "written.")
    parser.add_argument(
        "--test_txt_db", default='./pretrain/txt_db/vcr_test.db', type=str,
        help="The output directory where the model checkpoints will be "
             "written.")
    parser.add_argument(
        "--test_img_db", default='./pretrain/img_db/vcr_test;./pretrain/img_db/vcr_gt_test', type=str,
        help="The output directory where the model checkpoints will be "
             "written.")
    # Prepro parameters
    parser.add_argument('--max_txt_len', type=int, default=200,
                        help='max number of tokens in text (BERT BPE)')
    parser.add_argument('--conf_th', type=float, default=0.2,
                        help='threshold for dynamic bounding boxes '
                             '(-1 for fixed)')
    parser.add_argument('--max_bb', type=int, default=100,
                        help='max number of bounding boxes')
    parser.add_argument('--min_bb', type=int, default=10,
                        help='min number of bounding boxes')
    parser.add_argument('--num_bb', type=int, default=36,
                        help='static number of bounding boxes')

    # training parameters
    parser.add_argument("--num_sample_batch", default=1, type=int,
                        help="Total batch size for training. "
                             "(batch by tokens)")
    parser.add_argument("--train_batch_size", default=8196, type=int,
                        help="Total batch size for training. "
                             "(batch by tokens)")
    parser.add_argument("--val_batch_size", default=8196, type=int,
                        help="Total batch size for validation. "
                             "(batch by tokens)")
    parser.add_argument("--optim", default='adam',
                        choices=['adam', 'adamax', 'adamw'],
                        help="optimizer")
    parser.add_argument("--betas", default=[0.9, 0.98], nargs='+',
                        help="beta for adam optimizer")
    parser.add_argument("--dropout", default=0.1, type=float,
                        help="tune dropout regularization")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="weight decay (L2) regularization")
    parser.add_argument("--grad_norm", default=2.0, type=float,
                        help="gradient clipping (-1 for no clipping)")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Number of training steps to perform linear "
                             "learning rate warmup for. (invsqrt decay)")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead "
                             "of 32-bit")
    parser.add_argument('--pin_mem', action='store_true', help="pin memory")

    # can use config files
    parser.add_argument('--config', help='JSON config files')

    args = parse_with_config(parser)


    if args.conf_th == -1:
        assert args.max_bb + args.max_txt_len + 2 <= 512
    else:
        assert args.num_bb + args.max_txt_len + 2 <= 512
    n_gpu = args.num_gpu
    checkpoint = {}
    model_qa = UniterForVisualCommonsenseReasoning_inf.from_pretrained(
        args.encoder_config, checkpoint, img_dim=IMG_DIM)
    model_qa.init_type_embedding()
    model_qa.init_word_embedding(NUM_SPECIAL_TOKENS)
    LOGGER.info("***** Loading model from %s *****", args.model_qa_checkpoint)
    ckpt = torch.load(args.model_qa_checkpoint)
    model_qa.load_state_dict(ckpt['model_state_dict'])
    model_qa.cuda()
    model_qa.eval()

    model_qar = UniterForVisualCommonsenseReasoning_inf.from_pretrained(
        args.encoder_config, checkpoint, img_dim=IMG_DIM)
    model_qar.init_type_embedding()
    model_qar.init_word_embedding(NUM_SPECIAL_TOKENS)
    LOGGER.info("***** Loading model from %s *****", args.model_qar_checkpoint)
    ckpt = torch.load(args.model_qar_checkpoint)
    model_qar.load_state_dict(ckpt['model_state_dict'])
    model_qar.cuda()
    model_qar.eval()
    all_img_dbs = ImageLmdbGroup(args.conf_th, args.max_bb, args.min_bb,
                                 args.num_bb, args.compressed_db)
    if args.do_predict:
        test_img_db, test_img_db_gt = load_img_feat(
            args.test_img_db, all_img_dbs, args)
        test_txt_db = VcrTxtTokLmdb(args.test_txt_db, -1)
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
        test_final_dataset = Comet_qar_val_VcrDataset('test',tokenizer,test_txt_db, img_db=test_img_db, img_db_gt=test_img_db_gt)
        test_final_dataloader = build_dataloader(
            test_final_dataset, Comet_qar_val_vcr_collate,
            False, args)
        predict(model_qa, model_qar,test_final_dataloader,args)
    else:
        val_img_db, val_img_db_gt = load_img_feat(
            args.val_img_db, all_img_dbs, args)
        val_txt_db = VcrTxtTokLmdb(args.val_txt_db, -1)
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False,local_files_only=True)
        val_final_dataset = Comet_qar_val_VcrDataset('val',tokenizer, val_txt_db, img_db=val_img_db, img_db_gt=val_img_db_gt)
        val_final_dataloader = build_dataloader(
            val_final_dataset, Comet_qar_val_vcr_collate,
            False, args)
        validate(model_qa,model_qar, val_final_dataloader, args)
