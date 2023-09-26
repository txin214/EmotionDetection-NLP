import argparse
import json
import logging
import os
import pandas as pd
import numpy as np
import torch
from transformers import BertConfig, BertTokenizer, BertModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import BertNeutralClassifier
from simple_loader import load_and_cache_examples
from utils import *
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

def load_model(args):
    
    label_list = [0,1]
    config = BertConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=len(label_list),
        finetuning_task=args.task,
        id2label={str(i): label for i, label in enumerate(label_list)},
        label2id={label: i for i, label in enumerate(label_list)}
    )
    
    model_bert = BertModel(config=config)
    best_dir = os.path.join(args.output_dir, "current_best")
    filepath = os.path.join(best_dir, "model.pt")
    trained_state_dict = torch.load(filepath)
    
    bert_dict = {key.replace('bert.', ''): value for key, value in trained_state_dict["model"].items()}
    del bert_dict['classifier.weight']
    del bert_dict['classifier.bias']
    model_bert.load_state_dict(bert_dict)

    return model_bert

def get_dataloader(args, mode, tokenizer_dir):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)
    dataset = load_and_cache_examples(args, tokenizer, mode=mode, get_label=True)
    dataloader = DataLoader(dataset, batch_size=args.eval_batch_size)
    return dataloader

def get_feature(args, dataloader, model):
    features = None
    y = None
    for _, batch in enumerate(tqdm(dataloader)):
        model.eval()
        labels = batch[3].numpy()
        if y is None:
            y = labels
        else:
            y = np.concatenate((y, labels), axis=0)

        batch = tuple(t.to(args.device) for t in batch)
        
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2]
            }
            
            feature = model(**inputs).pooler_output.detach().cpu().numpy()
            if features is None:
                features = feature
            else:
                features = np.concatenate((features, feature), axis=0)
    return features, y



if __name__ == '__main__':
    config_filename = "original_neutral.json"
    with open(os.path.join("config", config_filename)) as f:
        args = Struct(**json.load(f))
    args.output_dir = os.path.join(args.ckpt_dir, args.output_dir)
    output_dir = os.path.join(args.output_dir, "visualization")
    init_logger()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_bert = load_model(args)
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    model_bert.to(args.device)
    modes = ["train", "dev", "test"]
    
    #features = {}
    best_dir = os.path.join(args.output_dir, "current_best")
    tokenizer_dir = os.path.join(best_dir, "tokenizer")
    features = {}
    y = {}
    
    for mode in modes:
        logger.info(f"Load {mode}-data.")
        dataloader = get_dataloader(args, mode, tokenizer_dir)
        features[mode], y[mode] = get_feature(args, dataloader, model_bert)
        print(y[mode].shape)
        np.savetxt(f"{mode}_feature.csv", features[mode], delimiter=",")
        np.savetxt(f"{mode}_labels.csv", y[mode], delimiter=",")
    

    

    