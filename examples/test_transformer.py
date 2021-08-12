from __future__ import print_function

import time
import dill as pickle
import itertools
from datetime import datetime, timedelta
import argparse
import os
import math
import sys
import warnings
import numpy as np
from distutils.version import LooseVersion
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
strhdlr = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s')
strhdlr.setFormatter(formatter)
logger.addHandler(strhdlr) 


import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torch.optim.lr_scheduler import LambdaLR
from torchvision import datasets, transforms
import horovod.torch as hvd
from tqdm import tqdm
from distutils.version import LooseVersion
import imagenet_resnet as models
from utils import *

# torchtext and transformer
from torchtext.data import Field, Dataset, BucketIterator
import transformer.Constants as Constants
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim

import kfac
os.environ['HOROVOD_NUM_NCCL_STREAMS'] = '1' 


def initialize():
    parser = argparse.ArgumentParser()

    # training settings
    parser.add_argument('-data_pkl', default=None)     # all-in-1 data pickle

    parser.add_argument('-epoch', type=int, default=200)
    parser.add_argument('-b', '--batch_size', type=int, default=256)

    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-warmup','--n_warmup_steps', type=int, default=4000)
    parser.add_argument('-lr_mul', type=float, default=2.0)
    parser.add_argument('-label_smoothing', action='store_true')

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')
    parser.add_argument('-scale_emb_or_prj', type=str, default='prj')

    parser.add_argument('-output_dir', type=str, default=None)
    parser.add_argument('-use_tb', action='store_true')
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    # KFAC Parameters
    parser.add_argument('--kfac-name', type=str, default='inverse',
            help='choises: %s' % kfac.kfac_mappers.keys() + ', default: '+'inverse')
    parser.add_argument('--exclude-parts', type=str, default='',
            help='choises: CommunicateInverse,ComputeInverse,CommunicateFactor,ComputeFactor')
    parser.add_argument('--kfac-update-freq', type=int, default=0,
                        help='iters between kfac inv ops (0 = no kfac) (default: 0)')
    parser.add_argument('--kfac-cov-update-freq', type=int, default=1,
                        help='iters between kfac cov ops (default: 1)')
    parser.add_argument('--kfac-update-freq-alpha', type=float, default=10,
                        help='KFAC update freq multiplier (default: 10)')
    parser.add_argument('--kfac-update-freq-decay', nargs='+', type=int, default=None,
                        help='KFAC update freq schedule (default None)')
    parser.add_argument('--stat-decay', type=float, default=0.95,
                        help='Alpha value for covariance accumulation (default: 0.95)')
    parser.add_argument('--damping', type=float, default=0.002,
                        help='KFAC damping factor (default 0.003)')
    parser.add_argument('--damping-alpha', type=float, default=0.5,
                        help='KFAC damping decay factor (default: 0.5)')
    parser.add_argument('--damping-decay', nargs='+', type=int, default=[40, 80],
                        help='KFAC damping decay schedule (default [40, 80])')
    parser.add_argument('--kl-clip', type=float, default=0.001,
                        help='KL clip (default: 0.001)')
    parser.add_argument('--diag-blocks', type=int, default=1,
                        help='Number of blocks to approx layer factor with (default: 1)')
    parser.add_argument('--diag-warmup', type=int, default=0,
                        help='Epoch to start diag block approximation at (default: 0)')
    parser.add_argument('--distribute-layer-factors', action='store_true', default=None,
                        help='Compute A and G for a single layer on different workers. '
                              'None to determine automatically based on worker and '
                              'layer count.')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    args = parser.parse_args()
    args.d_word_vec = args.d_model
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    hvd.init()
    torch.manual_seed(args.seed)
    args.verbose = 1 if hvd.rank() == 0 else 0

    if args.cuda:
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)

    cudnn.benchmark = True

    logfile = './logs/debug_multi30k_transformer_kfac{}_gpu{}_bs{}_{}.log'.format(args.kfac_update_freq, hvd.size(), args.batch_size, args.kfac_name)
    hdlr = logging.FileHandler(logfile)
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    if args.verbose:
        logger.info(args)

    if args.batch_size < 2048 and args.n_warmup_steps <= 4000:
        if args.verbose:
            logger.info('[Warning] The warmup steps may be not enough.\n'\
              '(sz_b, warmup) = (2048, 4000) is the official setting.\n'\
              'Using smaller batch w/o longer warmup may cause '\
              'the warmup stage ends with only little data trained.')

    return args


def prepare_dataloaders(args):
    batch_size = args.batch_size
    data = pickle.load(open(args.data_pkl, 'rb'))

    args.max_token_seq_len = data['settings'].max_len
    args.src_pad_idx = data['vocab']['src'].vocab.stoi[Constants.PAD_WORD]
    args.trg_pad_idx = data['vocab']['trg'].vocab.stoi[Constants.PAD_WORD]

    args.src_vocab_size = len(data['vocab']['src'].vocab)
    args.trg_vocab_size = len(data['vocab']['trg'].vocab)

    #========= Preparing Model =========#
    if args.embs_share_weight:
        assert data['vocab']['src'].vocab.stoi == data['vocab']['trg'].vocab.stoi, \
            'To sharing word embedding the src/trg word2idx table shall be the same.'

    fields = {'src': data['vocab']['src'], 'trg':data['vocab']['trg']}

    train = Dataset(examples=data['train'], fields=fields)
    val = Dataset(examples=data['valid'], fields=fields)

    train_iterator = BucketIterator(train, batch_size=batch_size, train=True)
    val_iterator = BucketIterator(val, batch_size=batch_size)

    # split train_iterator for distributed training
    divisible_size = math.ceil(len(train_iterator) / hvd.size()) * hvd.size()
    train_set = [next(itertools.cycle(train_iterator)) for _ in range(divisible_size)]
    train_subset = train_set[hvd.rank():divisible_size:hvd.size()]

    return train_subset, val_iterator

def patch_src(src, pad_idx):
    src = src.transpose(0, 1)
    return src

def patch_trg(trg, pad_idx):
    trg = trg.transpose(0, 1)
    trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
    return trg, gold

def get_model(args):
    model = Transformer(
        args.src_vocab_size,
        args.trg_vocab_size,
        src_pad_idx=args.src_pad_idx,
        trg_pad_idx=args.trg_pad_idx,
        trg_emb_prj_weight_sharing=args.proj_share_weight,
        emb_src_trg_weight_sharing=args.embs_share_weight,
        d_k=args.d_k,
        d_v=args.d_v,
        d_model=args.d_model,
        d_word_vec=args.d_word_vec,
        d_inner=args.d_inner_hid,
        n_layers=args.n_layers,
        n_head=args.n_head,
        dropout=args.dropout,
        scale_emb_or_prj=args.scale_emb_or_prj)

    if args.cuda:
        model.cuda()

    optimizer = ScheduledOptim(
        optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
        args.lr_mul, args.d_model, args.n_warmup_steps)

    if args.kfac_update_freq > 0:
        KFAC = kfac.get_kfac_module(args.kfac_name)
        preconditioner = KFAC(
                model, lr=args.base_lr, factor_decay=args.stat_decay,
                damping=args.damping, kl_clip=args.kl_clip,
                fac_update_freq=args.kfac_cov_update_freq,
                kfac_update_freq=args.kfac_update_freq,
                diag_blocks=args.diag_blocks,
                diag_warmup=args.diag_warmup,
                distribute_layer_factors=args.distribute_layer_factors, exclude_parts=args.exclude_parts)
        kfac_param_scheduler = kfac.KFACParamScheduler(
                preconditioner,
                damping_alpha=args.damping_alpha,
                damping_schedule=args.damping_decay,
                update_freq_alpha=args.kfac_update_freq_alpha,
                update_freq_schedule=args.kfac_update_freq_decay,
                start_epoch=args.resume_from_epoch)
    else:
        preconditioner = None

    optimizer = hvd.DistributedOptimizer(
            optimizer, named_parameters=model.named_parameters(),
            op=hvd.Average)

    if hvd.size() > 1:
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    return model, optimizer, preconditioner


def cal_performance(pred, gold, trg_pad_idx, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, trg_pad_idx, smoothing=smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word


def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
    return loss

def train(epoch, model, optimizer, preconditioner, train_subset, args):
    model.train()
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')
    
    avg_time = 0.0
    display = 1

    for batch_idx, batch in enumerate(train_subset):
        stime = time.time()

        # prepare data
        if args.cuda:
            src_seq = patch_src(batch.src, args.src_pad_idx).cuda()
            trg_seq, gold = map(lambda x: x.cuda(), patch_trg(batch.trg, args.trg_pad_idx))
        else:
            src_seq = patch_src(batch.src, args.src_pad_idx)
            trg_seq, gold = map(lambda x: x, patch_trg(batch.trg, args.trg_pad_idx))

        # forward
        optimizer.zero_grad()
        pred = model(src_seq, trg_seq)

        loss, n_correct, n_word = cal_performance(
            pred, gold, args.trg_pad_idx, smoothing=args.label_smoothing) 
        
        with torch.no_grad():
            train_loss.update(loss)
            train_accuracy.update(n_correct / n_word)

        # backward and update parameters
        loss.backward()

        optimizer.synchronize()

        if preconditioner is not None:
            preconditioner.step(epoch=epoch)

        with optimizer.skip_synchronize():
            optimizer.step()    

        avg_time += (time.time() - stime)

        # if (batch_idx + 1) % display == 0:
        #     if args.verbose:
        #         logger.info("[%d][%d] time: %.3f, speed: %.3f samples/s" % (epoch, batch_idx, avg_time/display, args.batch_size/(avg_time/display)))

    if args.verbose:
        logger.info("[%d] epoch train loss: %.4f, acc: %.3f" % (epoch, train_loss.avg.item(), 100*train_accuracy.avg.item()))


def validate(epoch, model, val_iterator, args):
    model.eval()
    total_loss, n_word_total, n_word_correct = 0, 0, 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_iterator):
            # prepare data
            if args.cuda:
                src_seq = patch_src(batch.src, args.src_pad_idx).cuda()
                trg_seq, gold = map(lambda x: x.cuda(), patch_trg(batch.trg, args.trg_pad_idx))
            else:
                src_seq = patch_src(batch.src, args.src_pad_idx)
                trg_seq, gold = map(lambda x: x, patch_trg(batch.trg, args.trg_pad_idx))

            # forward
            pred = model(src_seq, trg_seq)
            loss, n_correct, n_word = cal_performance(
                pred, gold, args.trg_pad_idx, smoothing=False)

            # note keeping
            n_word_total += n_word
            n_word_correct += n_correct
            total_loss += loss.item()         
     
    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total

    ppl = math.exp(min(loss_per_word, 100))

    if args.verbose:
        logger.info("[%d] epoch evaluation loss: %.4f, ppl: %.4f, acc: %.3f" % (epoch, loss_per_word, ppl, 100*accuracy))
    
    



def main():
    ''' 
    Usage:
    python train.py -data_pkl m30k_deen_shr.pkl -log m30k_deen_shr -embs_share_weight -proj_share_weight -label_smoothing -output_dir output -b 256 -warmup 128000
    '''

    torch.multiprocessing.set_start_method('spawn')
    args = initialize()

    train_subset, val_iterator = prepare_dataloaders(args)

    print(len(train_subset)) # test

    model, optimizer, preconditioner = get_model(args)
    
    if args.verbose:
        logger.info("MODEL: %s", args.model)    

    start = time.time()

    for epoch in range(args.epochs):
        train(epoch, model, optimizer, preconditioner, train_subset, args)
        validate(epoch, model, val_iterator, args)

    if args.verbose:
        logger.info("\nTraining time: %s", str(timedelta(seconds=time.time() - start)))