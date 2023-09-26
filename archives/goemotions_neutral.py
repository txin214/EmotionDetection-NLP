import argparse
import json
import logging
import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import (
    BertConfig,
    BertTokenizer,
    get_linear_schedule_with_warmup
)

from model import BertNeutralClassifier
from train_util import *
from utils import *
from simple_loader import load_and_cache_examples

import warnings
warnings.filterwarnings('ignore')

'''
Modified version of https://github.com/monologg/GoEmotions-pytorch/blob/master/run_goemotions.py
'''

logger = logging.getLogger(__name__)


def main():

    # Read from config file and make args
    config_filename = "original_neutral.json"
    with open(os.path.join("config", config_filename)) as f:
        args = Struct(**json.load(f))
    logger.info("Training/evaluation parameters {}".format(args))

    args.output_dir = os.path.join(args.ckpt_dir, args.output_dir)

    init_logger()
    set_seed(args)

    tokenizer = BertTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
    )

    label_list = [0, 1]

    config = BertConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=len(label_list),
        finetuning_task=args.task,
        id2label={str(i): label for i, label in enumerate(label_list)},
        label2id={label: i for i, label in enumerate(label_list)}
    )

    model = BertNeutralClassifier.from_pretrained(
        args.model_name_or_path,
        config=config
    )
    
    # Load dataset

    train_dataset = load_and_cache_examples(args, tokenizer, mode="train") if args.train_file else None
    dev_dataset = load_and_cache_examples(args, tokenizer, mode="dev") if args.dev_file else None
    test_dataset = load_and_cache_examples(args, tokenizer, mode="test") if args.test_file else None
        
    # GPU or CPU
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    model.to(args.device)

    if dev_dataset is None:
        args.evaluate_test_during_training = True  # If there is no dev dataset, only use test dataset

    if args.do_train:
        global_step, tr_loss = train(args, model, tokenizer, train_dataset, dev_dataset, test_dataset)
        logger.info(" global_step = {}, average loss = {}".format(global_step, tr_loss))

    results = {}
    if args.do_eval:
        checkpoints = list(
            os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + "pytorch_model.bin", recursive=True))
        )
        if not args.eval_all_checkpoints:
            checkpoints = checkpoints[-1:]
        else:
            logging.getLogger("transformers.configuration_utils").setLevel(logging.WARN)  # Reduce logging
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1]
            
            model = BertNeutralClassifier.from_pretrained(checkpoint)
            
            model.to(args.device)
            result = evaluate(args, model, test_dataset, mode="test", global_step=global_step)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

        output_eval_file = os.path.join(args.output_dir, "ckpt_eval_results.txt")
        with open(output_eval_file, "w") as f_w:
            for key in sorted(results.keys()):
                f_w.write("{} = {}\n".format(key, str(results[key])))
        
        best_dir = os.path.join(args.output_dir, "current_best")
        bestpath = os.path.join(best_dir, "model.pt")
        saved = torch.load(bestpath)
        model.load_state_dict(saved['model'])
        model.to(args.device)
        best_result = evaluate(args, model, test_dataset, mode="test", global_step=global_step)
        best_eval_file = os.path.join(best_dir, "best_eval_results.txt")
        with open(best_eval_file, "w") as f_w:
            for key in sorted(best_result.keys()):
                f_w.write("{} = {}\n".format(key, str(best_result[key])))

if __name__ == '__main__':
    main()
