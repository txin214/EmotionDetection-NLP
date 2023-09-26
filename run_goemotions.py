import argparse
import json
import logging
import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
#from attrdict import AttrDict
from transformers import (
    BertConfig,
    BertTokenizer,
    get_linear_schedule_with_warmup
)
from model import BertForMultiLabelClassification, BertMLCR, BertForMultiLabelClassificationCPCC, BertMultiLevelClassifier
from utils import *
from data_loader import (
    load_and_cache_examples,
    load_all,
    GoEmotionsProcessor
)
import warnings
warnings.filterwarnings('ignore')

'''
Modified version of https://github.com/monologg/GoEmotions-pytorch/blob/master/run_goemotions.py
'''

logger = logging.getLogger(__name__)

def save_model(model, args, filepath):
    save_info = {
        'model': model.state_dict(),
        'args': args
    }
    torch.save(save_info, filepath)
    logger.info("Saving model checkpoint to {}".format(filepath))

def train(args,
          model,
          tokenizer,
          train_dataset,
          dev_dataset=None,
          test_dataset=None,
          is_all=False):
          
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    # Use gradient accumulation trick to mimick larger batch size
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        # Default gradient_accumulation_steps is 1
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    '''
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, 
    after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    '''
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(t_total * args.warmup_proportion),
        num_training_steps=t_total
    )

    # Default model_name_or_path is "bert-base-cased"
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Logging steps = %d", args.logging_steps)
    logger.info("  Save steps = %d", args.save_steps)

    global_step = 0
    tr_loss = 0.0
    best_dev_f1 = 0

    optimizer.zero_grad() 
    for epoch in range(int(args.num_train_epochs)):
        for step, batch in enumerate(tqdm(train_dataloader, desc=f'train-{epoch}')):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            if not is_all:
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "labels": batch[3]
                }
            else:
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "labels_original": batch[3],
                    "labels_ekman": batch[4],
                    "labels_senti": batch[5]
                }
            outputs = model(**inputs)
            loss = outputs[0]

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    len(train_dataloader) <= args.gradient_accumulation_steps
                    and (step + 1) == len(train_dataloader)
            ):
                # Default "max_grad_norm": 1.0,
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    if args.evaluate_test_during_training:
                        eval_result = evaluate(args, model, test_dataset, "test", global_step, is_all)
                    else:
                        train_result = evaluate(args, model, train_dataset, "train", global_step, is_all)
                        eval_result = evaluate(args, model, dev_dataset, "dev", global_step, is_all)
                    
                    if eval_result["macro_f1"] > best_dev_f1:
                        best_dev_f1 = eval_result["macro_f1"]
                        # Save best current model checkpoint
                        output_dir = os.path.join(args.output_dir, "current_best")
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        filepath = os.path.join(output_dir, "model.pt")
                        save_model(model, args, filepath)
                        tokenizer_out = os.path.join(output_dir, "tokenizer")
                        if not os.path.exists(tokenizer_out):
                            os.makedirs(tokenizer_out)
                        tokenizer.save_pretrained(tokenizer_out)


                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    

                    if args.save_optimizer:
                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to {}".format(output_dir))

            if args.max_steps > 0 and global_step > args.max_steps:
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            break

    return global_step, tr_loss / global_step


def evaluate(args, model, eval_dataset, mode, global_step=None, is_all=False):
    results = {}
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    if global_step != None:
        logger.info("***** Running evaluation on {} dataset ({} step) *****".format(mode, global_step))
    else:
        logger.info("***** Running evaluation on {} dataset *****".format(mode))
    logger.info("  Num examples = {}".format(len(eval_dataset)))
    logger.info("  Eval Batch size = {}".format(args.eval_batch_size))
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    logger.info("  Sit tight and wait, evaluation might take a while...")
    for _, batch in enumerate(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            if not is_all:
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "labels": batch[3]
                }
            else:
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "labels_original": batch[3],
                    "labels_ekman": batch[4],
                    "labels_senti": batch[5]
                }
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = 1 / (1 + np.exp(-logits.detach().cpu().numpy()))  # Sigmoid
            if not is_all:
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                out_label_ids = inputs["labels_original"].detach().cpu().numpy()
        else:
            preds = np.append(preds, 1 / (1 + np.exp(-logits.detach().cpu().numpy())), axis=0)  # Sigmoid
            if not is_all:
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
            else:
                out_label_ids = np.append(out_label_ids, inputs["labels_original"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    results = {
        "loss": eval_loss
    }
    # Multi-label classification, default threshold 0.3
    preds[preds > args.threshold] = 1
    preds[preds <= args.threshold] = 0
    result = compute_metrics(out_label_ids, preds)
    results.update(result)

    output_dir = os.path.join(args.output_dir, mode)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_eval_file = os.path.join(output_dir, "{}-{}.txt".format(mode, global_step) if global_step else "{}.txt".format(mode))
    with open(output_eval_file, "w") as f_w:
        logger.info("***** Eval results on {} dataset *****".format(mode))
        for key in sorted(results.keys()):
            logger.info("  {} = {}".format(key, str(results[key])))
            f_w.write("  {} = {}\n".format(key, str(results[key])))

    return results


def main(cli_args):
    # Read from config file and make args
    config_filename = "{}.json".format(cli_args.taxonomy)
    with open(os.path.join("config", config_filename)) as f:
        args = Struct(**json.load(f))
    logger.info("Training/evaluation parameters {}".format(args))

    args.output_dir = os.path.join(args.ckpt_dir, args.output_dir)
    if args.use_cpcc == True:
        args.output_dir = os.path.join(args.output_dir, "cpcc")
    if args.reg == True:
        args.output_dir = os.path.join(args.output_dir, "reg")
        
    init_logger()
    set_seed(args)

    tokenizer = BertTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
    )
    is_all = cli_args.taxonomy == "all"
    if not is_all:
        processor = GoEmotionsProcessor(args)
        label_list = processor.get_labels()
        config = BertConfig.from_pretrained(
            args.model_name_or_path,
            num_labels=len(label_list),
            finetuning_task=args.task,
            id2label={str(i): label for i, label in enumerate(label_list)},
            label2id={label: i for i, label in enumerate(label_list)}
        )
        if args.reg == True:
            model = BertMLCR.from_pretrained(
                args.model_name_or_path,
                config=config
            )
        if args.use_cpcc == False:
            model = BertForMultiLabelClassification.from_pretrained(
                args.model_name_or_path,
                config=config
            )
        else:
            class EmotionLabel:
                def __init__(self, label, ekman, sentiment):
                    self.label = label
                    self.sentiment = sentiment
                    self.ekman = ekman
            
            #### For calculating CPCC. Reference: https://openreview.net/pdf?id=7J-30ilaUZM
            sentiment_map = {"positive":["joy"], 
                            "negative":["anger", "disgust", "fear", "sadness"], 
                            "ambiguous":["surprise"], "neutral":["neutral"]}
            ekman_map = {"joy":["admiration", "amusement", "approval", "caring", "desire", "excitement", 
                                "gratitude", "joy", "love", "optimism", "pride", "relief"],
                        "anger":["anger", "annoyance", "disapproval"],
                        "disgust":["disgust"],
                        "fear":["fear", "nervousness"],
                        "sadness":["sadness", "disappointment", "embarrassment", "grief", "remorse"],
                        "surprise":["confusion", "curiosity", "surprise", "realization"],
                        "neutral":["neutral"]}

            EmotionLabelIds = {}
            label2id = {label: i for i, label in enumerate(label_list)}
            reversed_ekman_map = {value: key for key, values in ekman_map.items() for value in values}
            reversed_senti_map = {value: key for key, values in sentiment_map.items() for value in values}
            for key, value in reversed_ekman_map.items():
                emo_label = EmotionLabel(key, value, reversed_senti_map[value])
                EmotionLabelIds[label2id[key]] = emo_label
            t_distance = np.zeros((len(label_list), len(label_list)))
            for i1, label1 in EmotionLabelIds.items():
                for i2, label2 in EmotionLabelIds.items():
                    t_distance[i1, i2] = tree_metric(label1, label2)

            model = BertForMultiLabelClassificationCPCC.from_pretrained(
                args.model_name_or_path,
                config=config,
                t_distance=t_distance,
                threshold=args.threshold
            )
        # Load dataset
        train_dataset = load_and_cache_examples(args, tokenizer, mode="train") if args.train_file else None
        dev_dataset = load_and_cache_examples(args, tokenizer, mode="dev") if args.dev_file else None
        test_dataset = load_and_cache_examples(args, tokenizer, mode="test") if args.test_file else None

    else:
        config = BertConfig.from_pretrained(
            args.model_name_or_path,
            finetuning_task=args.task,
        )

        model = BertMultiLevelClassifier.from_pretrained(
                args.model_name_or_path,
                config=config
            )
        train_dataset = load_all(args, tokenizer, mode="train") if args.train_file else None
        dev_dataset = load_all(args, tokenizer, mode="dev") if args.dev_file else None
        test_dataset = load_all(args, tokenizer, mode="test") if args.test_file else None
    
        

    # GPU or CPU
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    model.to(args.device)

    

    if dev_dataset is None:
        args.evaluate_test_during_training = True  # If there is no dev dataset, only use test dataset

    if args.do_train:
        global_step, tr_loss = train(args, model, tokenizer, train_dataset, dev_dataset, test_dataset, is_all)
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
            if not is_all:
                if args.use_cpcc == False:
                    model = BertForMultiLabelClassification.from_pretrained(checkpoint)
                else:
                    model = BertForMultiLabelClassificationCPCC.from_pretrained(checkpoint)
            else:
                model = BertMultiLevelClassifier.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, test_dataset, mode="test", global_step=global_step, is_all=is_all)
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
        best_result = evaluate(args, model, test_dataset, mode="test", global_step=global_step, is_all=is_all)
        best_eval_file = os.path.join(best_dir, "best_eval_results.txt")
        with open(best_eval_file, "w") as f_w:
            for key in sorted(best_result.keys()):
                f_w.write("{} = {}\n".format(key, str(best_result[key])))




if __name__ == '__main__':
    
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument("--taxonomy", type=str, required=True, help="Taxonomy (original, ekman, group, all)")
    cli_args = cli_parser.parse_args()
    main(cli_args)
