# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for question-answering on SQuAD (DistilBERT, Bert, XLM, XLNet)."""


import argparse
import os
import random
import math
import timeit
import time
import datetime
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s')

strhdlr = logging.StreamHandler()
strhdlr.setFormatter(formatter)
logger.addHandler(strhdlr) 

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# for bert
from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features,
)
from transformers.data.metrics.squad_metrics import (
    compute_predictions_logits,
    squad_evaluate,
)
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor

MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

import horovod.torch as hvd
from utils import *

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def initialize():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_type",
        default='bert',
        type=str,
        help="Model type selected in the list: " + ", ".join(MODEL_TYPES),
    )
    parser.add_argument(
        "--model_name_or_path",
        default='bert-base-uncased',
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--log_dir",
        default='./logs',
        type=str
    )

    # Other parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="The input data dir. Should contain the .json files for the task."
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--predict_file",
        default=None,
        type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )

    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, the SQuAD examples contain some that do not have an answer.",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="If null_score - best_non_null is greater than the threshold predict null.",
    )

    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
        "be truncated to this length.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
    )
    parser.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        help="The maximum length of an answer that can be generated. This is needed because the start "
        "and end predictions are not conditioned on one another.",
    )
    parser.add_argument(
        "--verbose_logging",
        action="store_true",
        help="If true, all of the warnings related to data processing will be printed. "
        "A number of warnings are expected for a normal SQuAD evaluation.",
    )
    parser.add_argument(
        "--lang_id",
        default=0,
        type=int,
        help="language id of input for language-specific xlm models (see tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)",
    )

    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")

    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")
    args = parser.parse_args()

    if args.doc_stride >= args.max_seq_length - args.max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )


    # Setup CUDA, GPU & distributed training
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    hvd.init()

    args.local_rank = hvd.local_rank()

    # Set seed
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.set_device(args.local_rank)
        torch.cuda.manual_seed(args.seed)

    torch.backends.cudnn.benchmark = True    

    # Logging Settings
    os.makedirs(args.log_dir, exist_ok=True)
    logfile = os.path.join(args.log_dir,
        '{}_ep{}_bs{}_gpu{}.log'.format(args.model_type, args.num_train_epochs, args.per_gpu_train_batch_size, hvd.comm.size()))

    hdlr = logging.FileHandler(logfile)
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 

    args.verbose = True if hvd.rank() == 0 else False
    
    if args.verbose:
        logger.info(args)
    
    return args

def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):
    if hvd.rank() > 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        hvd.barrier()

    # Load data features from cache or dataset file
    input_dir = args.data_dir if args.data_dir else "."
    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
        ),
    )

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        if hvd.rank() == 0 and not evaluate:
            # Make sure only the first process in distributed training process the dataset, and the others will use the cache
            hvd.barrier()
        logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        logger.info("Creating features from dataset file at %s", input_dir)

        if not args.data_dir and ((evaluate and not args.predict_file) or (not evaluate and not args.train_file)):
            try:
                import tensorflow_datasets as tfds
            except ImportError:
                raise ImportError("If not data_dir is specified, tensorflow_datasets needs to be installed.")

            if args.version_2_with_negative:
                logger.warn("tensorflow_datasets does not handle version 2 of SQuAD.")

            tfds_examples = tfds.load("squad")
            examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=evaluate)
        else:
            processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()
            if evaluate:
                examples = processor.get_dev_examples(args.data_dir, filename=args.predict_file)
            else:
                examples = processor.get_train_examples(args.data_dir, filename=args.train_file)

        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
            return_dataset="pt",
            threads=args.threads,
        )

        if hvd.rank() == 0:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

        if hvd.rank() == 0 and not evaluate:
            # Make sure only the first process in distributed training process the dataset, and the others will use the cache
            torch.distributed.barrier()

    if output_examples:
        return dataset, examples, features
    return dataset

def build_tokenizer(args):
    if hvd.size() > 1 and hvd.rank() > 0:
        hvd.barrier()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if hvd.size() > 1 and hvd.rank() == 0:
        hvd.barrier()
    return tokenizer

def get_dataset(args):
    tokenizer = build_tokenizer(args)

    train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False)

    eval_dataset, eval_examples, eval_features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)
    test_inputs = (eval_examples, eval_features, tokenizer) # used to calculate F1 score

    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    args.batch_size = args.per_gpu_train_batch_size

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size, sampler=train_sampler, **kwargs)

    args.eval_batch_size = args.per_gpu_eval_batch_size
    test_sampler = torch.utils.data.SequentialSampler(eval_dataset)
    test_loader = torch.utils.data.DataLoader(eval_dataset,batch_size=args.eval_batch_size, sampler=test_sampler, **kwargs)

    return train_sampler, train_loader, test_loader, test_inputs

def get_model(args):
    # Load pretrained model
    if hvd.size() > 1 and hvd.rank() > 0:
        hvd.barrier()
    
    args.model_type = args.model_type.lower()
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    model = AutoModelForQuestionAnswering.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    
    if hvd.rank() == 0 and hvd.size() > 1:
        hvd.barrier()
    
    if args.cuda:
        model.cuda()

    # optimizer
    criterion = nn.CrossEntropyLoss()

    # K-FAC
    preconditioner = None

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    optimizer = hvd.DistributedOptimizer(optimizer, 
                                        named_parameters=model.named_parameters(),
                                        op=hvd.Average,
                                        backward_passes_per_step=1)
    if hvd.size() > 1:
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    return model, optimizer, preconditioner

def train(args, epoch, model, optimizer, preconditioner, train_sampler, train_loader, evaluate, test_loader):
    model.train()
    train_sampler.set_epoch(epoch)

    train_loss = Metric('train_loss')
    display = 50

    for batch_idx, batch in enumerate(train_loader):
        if args.cuda:
            batch = tuple(t.cuda() for t in batch)
        optimizer.zero_grad()
        
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
            "start_positions": batch[3],
            "end_positions": batch[4],
        }

        outputs = model(**inputs)
        loss = outputs[0]

        with torch.no_grad():
            train_loss.update(loss)
        
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % display == 0:
            if args.verbose:
                logger.info("[%d] epoch [%d] iteration train loss: %.4f" % (epoch, batch_idx, train_loss.avg.item()))
            if evaluate and hvd.rank() == 0: 
                eval_loss = cal_evaluate_loss(model, test_loader, args)
                logger.info("[%d] epoch [%d] iteration eval loss: %.4f" % (epoch, batch_idx, eval_loss))


def cal_evaluate_loss(model, test_loader, args):
    model.eval()
    loss_sum = 0.0
    loss_cnt = 0

    for batch in test_loader:
        if args.cuda:
            batch = tuple(t.cuda() for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "return_dict": False,
            }

            feature_indices = batch[3]
            outputs = model(**inputs)
            loss = outputs[0]

            loss_sum += loss.item()
            loss_cnt += batch[0].shape[0]

    model.train()
    return loss_sum / loss_cnt


def cal_evaluate_F1_score(model, test_loader, test_inputs, args):
    model.eval()
    examples, features, tokenizer = test_inputs[0], test_inputs[1], test_inputs[2]

    all_results = []
    for batch in test_loader:
        if args.cuda:
            batch = tuple(t.cuda() for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "return_dict": False,
            }

            feature_indices = batch[3]

            outputs = model(**inputs)

        for i, feature_index in enumerate(feature_indices):
            eval_feature = features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [to_list(output[i]) for output in outputs]

            # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
            # models only use two.
            if len(output) >= 5:
                start_logits = output[0]
                start_top_index = output[1]
                end_logits = output[2]
                end_top_index = output[3]
                cls_logits = output[4]

                result = SquadResult(
                    unique_id,
                    start_logits,
                    end_logits,
                    start_top_index=start_top_index,
                    end_top_index=end_top_index,
                    cls_logits=cls_logits,
                )

            else:
                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)

    
    predictions = compute_predictions_logits(
        examples,
        features,
        all_results,
        args.n_best_size,
        args.max_answer_length,
        args.do_lower_case,
        None, #output_prediction_file
        None, #output_nbest_file
        None, #output_null_log_odds_file
        args.verbose_logging,
        args.version_2_with_negative,
        args.null_score_diff_threshold,
        tokenizer,
    )

    # Compute the F1 and exact scores.
    result = squad_evaluate(examples, predictions)
    return result

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')

    args = initialize()

    train_sampler, train_loader, test_loader, test_inputs = get_dataset(args)

    model, optimizer, preconditioner = get_model(args)
    eval_in_progress = True

    start = time.time()
    for epoch in args.num_train_epochs:
        train(args, epoch, model, optimizer, preconditioner, train_sampler, train_loader, eval_in_progress, test_loader)

    if args.verbose:
        logger.info("Total Training Time: %s", str(datetime.timedelta(seconds=time.time() - start)))
    
    eval_F1_score = True
    if eval_F1_score and hvd.rank() == 0:
        result = cal_evaluate_F1_score(model, test_loader, test_inputs, args)
        logger.info("Evaluation result: {}".format(result))