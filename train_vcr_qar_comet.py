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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from torch.utils.data import DataLoader

from transformers.models.bert import BertTokenizer
from data import ( DetectFeatLmdb,
                  VcrTxtTokLmdb, ImageLmdbGroup )
from model.vcr_caption import UniterForVisualCommonsenseReasoning
from data.vcr_comet import Comet_qar_VcrDataset,Comet_qar_vcr_collate,Comet_qar_val_VcrDataset,Comet_qar_val_vcr_collate

from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from utils.misc import  parse_with_config, set_dropout
from utils.const import  IMG_DIM
NUM_SPECIAL_TOKENS = 81
from progressbar import ProgressBar


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


def main(opts):
    n_gpu =opts.num_gpu
    if opts.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, "
                         "should be >= 1".format(
                            opts.gradient_accumulation_steps))

    # load DBs and image dirs
    all_img_dbs = ImageLmdbGroup(opts.conf_th, opts.max_bb, opts.min_bb,
                                 opts.num_bb, opts.compressed_db)
    # train
    LOGGER.info(f"Loading Train Dataset "
                f"{opts.train_txt_dbs}, {opts.train_img_dbs}")
    img_db, img_db_gt = load_img_feat(opts.train_img_dbs, all_img_dbs, opts)
    qa_txt_db = VcrTxtTokLmdb(opts.train_txt_dbs, opts.max_txt_len, task="qa")
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)


    train_dataset=Comet_qar_VcrDataset(tokenizer,qa_txt_db, img_db_gt=img_db_gt, img_db=img_db)
    train_dataloader = build_dataloader(train_dataset, Comet_qar_vcr_collate, True, opts)

    LOGGER.info(f"Loading Val Dataset {opts.val_txt_db}, {opts.val_img_db}")
    val_img_db, val_img_db_gt = load_img_feat(
        opts.val_img_db, all_img_dbs, opts)
    val_txt_db = VcrTxtTokLmdb(opts.val_txt_db, -1)
    val_dataset = Comet_qar_val_VcrDataset(tokenizer,val_txt_db, img_db=val_img_db, img_db_gt=val_img_db_gt)
    val_dataloader = build_dataloader(val_dataset, Comet_qar_val_vcr_collate,
                                      False, opts)
    checkpoint = torch.load(opts.encoder_checkpoint)

    all_dbs = [opts.train_txt_dbs] + [opts.val_txt_db]
    toker = json.load(open(f'{all_dbs[0]}/meta.json'))['bert']
    assert all(toker == json.load(open(f'{db}/meta.json'))['bert']
               for db in all_dbs)

    model_encoder = UniterForVisualCommonsenseReasoning.from_pretrained(opts.encoder_config, checkpoint, img_dim=IMG_DIM)
    model_encoder.init_type_embedding()
    model_encoder.init_word_embedding(NUM_SPECIAL_TOKENS)
    # optimizer=torch.optim.Adam()

    n_epoch = 0
    optimizer = torch.optim.Adam(model_encoder.parameters(), lr=opts.learning_rate, eps=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    if opts.global_step!=0:
        ckpt = torch.load(opts.model_checkpoint)
        model_encoder.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()  # an optimizer.cuda() method for this operation would be nice
        LOGGER.info("***** Loading checkpoint from %s *****", opts.model_checkpoint)
        n_epoch = ckpt['epoch']


    model_encoder.cuda()


    # make sure every process has same model parameters in the beginning
    set_dropout(model_encoder, opts.dropout)

    global_step = opts.global_step

    TB_LOGGER.create(join(opts.output_dir, 'log'))
    if not exists(join(opts.output_dir, 'results')):
        os.makedirs(join(opts.output_dir, 'results'))
    if not exists(join(opts.output_dir, 'ckpt')):
        os.makedirs(join(opts.output_dir, 'ckpt'))
    add_log_to_file(join(opts.output_dir, 'log', 'log.txt'))


    LOGGER.info(f"***** Running training with {n_gpu} GPUs *****")
    LOGGER.info("  Num examples = %d", len(train_dataset))
    LOGGER.info("  Batch size = %d", opts.num_sample_batch)
    LOGGER.info("  Accumulate steps = %d", opts.gradient_accumulation_steps)
    LOGGER.info("  Epoch = %d", opts.epoch)

    model_encoder.train()
    n_examples = 0

    start = time()
    # quick hack for amp delay_unscale bug
    optimizer.zero_grad()
    tr_loss = 0.0
    tr_cls_loss=0.0
    tr_match_loss=0.0
    n_correct_qa=0
    total_step=0
    while n_epoch<opts.epoch:
        print(f"Epoch {n_epoch}/{opts.epoch}")
        pbar = ProgressBar(n_total=len(train_dataset)//opts.num_sample_batch, desc='training')

        for step, batch in enumerate(train_dataloader):
            n_examples +=opts.num_sample_batch
            rank_scores,loss,matched = model_encoder(batch)
            n_correct_qa += matched.sum().item()
            global_step += 1
            lr_this_step = optimizer.param_groups[0]['lr']
            TB_LOGGER.add_scalar('lr', lr_this_step, global_step)
            TB_LOGGER.add_scalar('loss', loss.item(), global_step)
            TB_LOGGER.step()
            loss.backward()
            if total_step % opts.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            tr_loss += loss.item()

            total_step += 1
            pbar(step=step, info={'loss': tr_loss / total_step,'a_acc':n_correct_qa/n_examples})

            if global_step%20000==0:
                ckpt_dir=os.path.join(opts.output_dir,'ckpt')
                output_dir = os.path.join(ckpt_dir,
                                          "UNITER-checkpoint-{}-.tar".format(global_step))
                torch.save({
                    'epoch': n_epoch,
                    'model_state_dict': model_encoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'loss': loss,
                }, output_dir)
                LOGGER.info("Saving model checkpoint to %s", output_dir)

            if global_step % 100 == 0:
                # monitor training throughput
                LOGGER.info(f'============Step {global_step}=============')
                tot_ex = n_examples
                ex_per_sec = int(tot_ex / (time()-start))
                LOGGER.info(f'{tot_ex} examples trained at '
                            f'{ex_per_sec} ex/s')
                TB_LOGGER.add_scalar('perf/ex_per_s',
                                     ex_per_sec, global_step)
                LOGGER.info('===========================================')

            if global_step % opts.valid_steps == 0 and n_epoch>=opts.epoch_begin:
                score = validate(model_encoder, val_dataloader,opts)
                #TB_LOGGER.log_scaler_dict(val_acc)
                ckpt_dir = os.path.join(opts.output_dir, 'ckpt')
                output_dir = os.path.join(ckpt_dir,
                                          "UNITER-checkpoint-{}-{:4f}.tar".format(global_step,score))
                torch.save({
                    'epoch': n_epoch,
                    'model_state_dict': model_encoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'loss': loss,
                }, output_dir)
                LOGGER.info("Saving model checkpoint to %s", output_dir)
                model_encoder.train()
        scheduler.step()
        n_epoch += 1
        LOGGER.info(f"finished {n_epoch} epochs")
    if global_step % opts.valid_steps != 0:
        score = validate(model_encoder, val_dataloader, opts)
        # TB_LOGGER.log_scaler_dict(val_acc)
        ckpt_dir = os.path.join(opts.output_dir, 'ckpt')
        output_dir = os.path.join(ckpt_dir,
                                  "UNITER-checkpoint-{}-{:4f}.tar".format(global_step, score))
        torch.save({
            'epoch': n_epoch,
            'model_state_dict': model_encoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'loss': loss,
        }, output_dir)
        LOGGER.info("Saving model checkpoint to %s", output_dir)


@torch.no_grad()
def validate(model, val_loader,args):
    val_pbar=ProgressBar(n_total=len(val_loader),desc='eval')
    LOGGER.info("start running validation...")
    model.eval()
    total_ex=0
    n_correct_qa=0

    for i, batch in enumerate(val_loader):
        total_ex += args.num_sample_batch
        with torch.no_grad():
            rank_scores,loss,matched_qa= model(batch)
        n_correct_qa += matched_qa.sum().item()
        val_pbar(step=i, info={'a_acc':n_correct_qa/total_ex})
    return n_correct_qa


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
    parser.add_argument("--model_checkpoint",
                        default='./outputs/QR_base/ckpt/UNITER-GPT-checkpoint-60000-.tar', type=str,
                        help="pretrained model")
    parser.add_argument("--encoder_checkpoint",
                        default='./pretrain/pretrained/uniter-large.pt', type=str,
                        help="pretrained model")

    parser.add_argument("--optimizer_checkpoint",
                        default=None, type=str,
                        help="pretrained model")

    parser.add_argument("--task",
                        default='qa', type=str,
                        choices=['qa', 'qar','qra','qr'],
                        help="which setting is checkpoint from")

    parser.add_argument(
        "--output_dir", default='./outputs/qar_comet', type=str,
        help="The output directory where the model checkpoints will be "
             "written.")
    parser.add_argument(
        "--global_step", default=0, type=int,
        help="")
    parser.add_argument(
        "--train_txt_dbs", default='./pretrain/txt_db/vcr_train.db', type=str,
        help="The output directory where the model checkpoints will be "
             "written.")
    parser.add_argument(
        "--train_img_dbs", default='./pretrain/img_db/vcr_train;./pretrain/img_db/vcr_gt_train', type=str,
        help="The output directory where the model checkpoints will be "
             "written.")
    parser.add_argument(
        "--val_txt_db", default='./pretrain/txt_db/vcr_val.db', type=str,
        help="The output directory where the model checkpoints will be "
             "written.")
    parser.add_argument(
        "--val_img_db", default='./pretrain/img_db/vcr_val;./pretrain/img_db/vcr_gt_val', type=str,
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
    parser.add_argument("--num_sample_batch", default=4, type=int,
                        help="Total batch size for training. "
                             "(batch by tokens)")
    parser.add_argument("--train_batch_size", default=8196, type=int,
                        help="Total batch size for training. "
                             "(batch by tokens)")
    parser.add_argument("--val_batch_size", default=8196, type=int,
                        help="Total batch size for validation. "
                             "(batch by tokens)")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                        help="Number of updates steps to accumualte before "
                             "performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--lr_mul", default=10.0, type=float,
                        help="multiplier for top layer lr")
    parser.add_argument("--epoch_begin", default=3, type=int)
    parser.add_argument("--valid_steps", default=5000, type=int,
                        help="Run validation begin")
    parser.add_argument("--epoch", default=100, type=int,
                        help="Total epoch.")
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

    # device parameters
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead "
                             "of 32-bit")
    parser.add_argument('--n_workers', type=int, default=0,
                        help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true', help="pin memory")

    # can use config files
    parser.add_argument('--config', help='JSON config files')

    args = parse_with_config(parser)

    # if exists(args.output_dir) and os.listdir(args.output_dir):
    #     raise ValueError("Output directory ({}) already exists and is not "
    #                      "empty.".format(args.output_dir))

    # options safe guard
    if args.conf_th == -1:
        assert args.max_bb + args.max_txt_len + 2 <= 512
    else:
        assert args.num_bb + args.max_txt_len + 2 <= 512

    if args.do_predict:
        n_gpu = args.num_gpu
        checkpoint={}
        model_encoder = UniterForVisualCommonsenseReasoning.from_pretrained(
            args.encoder_config, checkpoint, img_dim=IMG_DIM)
        model_encoder.init_type_embedding()
        model_encoder.init_word_embedding(NUM_SPECIAL_TOKENS)
        LOGGER.info("***** Loading model from %s *****", args.model_checkpoint)
        ckpt = torch.load(args.model_checkpoint)
        model_encoder.load_state_dict(ckpt['model_state_dict'])
        model_encoder.cuda()
        model_encoder.eval()
        all_img_dbs = ImageLmdbGroup(args.conf_th, args.max_bb, args.min_bb,
                                     args.num_bb, args.compressed_db)
        val_img_db, val_img_db_gt = load_img_feat(
            args.val_img_db, all_img_dbs, args)
        val_txt_db = VcrTxtTokLmdb(args.val_txt_db, -1)
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
        val_final_dataset = Comet_qar_val_VcrDataset(tokenizer,val_txt_db, img_db=val_img_db, img_db_gt=val_img_db_gt)
        val_final_dataloader = build_dataloader(
            val_final_dataset, Comet_qar_val_vcr_collate,
            False, args)
        validate(model_encoder, val_final_dataloader,args)
    else:
        main(args)
