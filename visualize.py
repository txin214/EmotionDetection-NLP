import argparse
import json
import logging
from sklearn.manifold import TSNE
import os
import pandas as pd
import numpy as np
import torch
#from attrdict import AttrDict
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from seaborn import clustermap
from sklearn.cluster import SpectralClustering
from transformers import BertConfig, BertTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import BertForMultiLabelClassification, BertMultiLevelClassifier, BertForMultiLabelClassificationCPCC
from data_loader import (
    load_and_cache_examples,
    load_all,
    GoEmotionsProcessor
)
from utils import *
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay

""" 
####### Original Code ######## 

*** Idea for label embedding ***

When we perform classification, we will pass the hidden representation through one last linear layer before calculating the loss.
We would like to interpret this last linear layer as the embedding layer of the labels. We explore this idea using some clusterings 
and visualizations here.  

The results provide support to this idea. One approach to inject prior knowledge of label similarity into the learning procedure 
is to impose regularizations on this final linear layer. 
"""

logger = logging.getLogger(__name__)

def get_args(cli_args):
    config_filename = "{}.json".format(cli_args.taxonomy)
    with open(os.path.join("config", config_filename)) as f:
        args = Struct(**json.load(f))

    args.output_dir = os.path.join(args.ckpt_dir, args.output_dir)
    if args.reg == True:
        args.output_dir = os.path.join(args.output_dir, "reg")
    elif args.use_cpcc == True:
        args.output_dir = os.path.join(args.output_dir, "cpcc")

    args.is_all = cli_args.taxonomy == "all"
    return args

def load_model(args, label_list):
    if not args.is_all:
        config = BertConfig.from_pretrained(
            args.model_name_or_path,
            num_labels=len(label_list),
            finetuning_task=args.task,
            id2label={str(i): label for i, label in enumerate(label_list)},
            label2id={label: i for i, label in enumerate(label_list)}
        )
        if args.use_cpcc == False:
            model = BertForMultiLabelClassification(config=config)
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
            model = BertForMultiLabelClassificationCPCC(config=config, t_distance=t_distance, threshold=args.threshold)
    else:
        config = BertConfig.from_pretrained(
            args.model_name_or_path,
            finetuning_task=args.task,
        )
        model = BertMultiLevelClassifier(config=config)

    best_dir = os.path.join(args.output_dir, "current_best")
    filepath = os.path.join(best_dir, "model.pt")
    saved = torch.load(filepath)
    model.load_state_dict(saved['model'])
    return model

def get_labels(args):
    processor = GoEmotionsProcessor(args)
    label_list = processor.get_labels()
    return label_list

def visualize_last_weight(args, model):
    if not args.is_all:
        classifier_weights = model.classifier.weight.data.numpy()
    else:
        classifier_weights = model.classifier_original.weight.data.numpy()
    return classifier_weights

def get_tsn_plot(label_embed, labels, filepath='layer_embed.png'):
    tsne = TSNE(n_components=2, random_state=0, perplexity=5)
    layer_embed_tsn = tsne.fit_transform(classifier_weights)
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(labels):
        x, y = layer_embed_tsn[i, :]
        plt.scatter(x, y, color="blue")
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords="offset points", ha="right", va="bottom")
    plt.savefig(filepath)

def get_cluster_plot(dot_sim_df):
    clustermap(dot_sim_df)
    plt.savefig("cluster.png")

def get_dataloader(args, mode, tokenizer_dir):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)
    if args.is_all == False:
        dataset = load_and_cache_examples(args, tokenizer, mode=mode)
    else:
        dataset = load_all(args, tokenizer, mode=mode)
    dataloader = DataLoader(dataset, batch_size=args.eval_batch_size)
    return dataloader

def get_pred_and_true(args, dataloader, model):
    y_preds = None
    y_true = None
    for _, batch in enumerate(tqdm(dataloader)):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            if not args.is_all:
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
            logits = outputs[1]

        if y_preds is None:
            y_preds = 1 / (1 + np.exp(-logits.detach().cpu().numpy()))  # Sigmoid
            if not args.is_all:
                y_true = inputs["labels"].detach().cpu().numpy()
            else:
                y_true = inputs["labels_original"].detach().cpu().numpy()
        else:
            y_preds = np.append(y_preds, 1 / (1 + np.exp(-logits.detach().cpu().numpy())), axis=0)  # Sigmoid
            if not args.is_all:
                y_true = np.append(y_true, inputs["labels"].detach().cpu().numpy(), axis=0)
            else:
                y_true = np.append(y_true, inputs["labels_original"].detach().cpu().numpy(), axis=0)

    y_preds[y_preds > args.threshold] = 1
    y_preds[y_preds <= args.threshold] = 0
    return y_preds, y_true

def get_confusion_table(y_preds, y_true, labels, mode, filepath):
    eps = 1e-9
    cm = multilabel_confusion_matrix(y_true, y_preds)
    tp = cm[:, 1, 1]
    fp = cm[:, 0, 1]
    tn = cm[:, 0, 0]
    fn = cm[:, 1, 0]
    precision = tp/(tp+fp+eps)
    recall = tp/(tp+fn+eps)
    f1 = 2*precision*recall/(precision+recall+eps)
    df = pd.DataFrame({"Emotion":labels, "Precision":precision, "Recall":recall, "F1": f1})
    df.to_csv(filepath, index=False, float_format='%.3f')

def plot_confusion_matrix(y_preds, y_true, labels, mode, filepath):
    cm = multilabel_confusion_matrix(y_true, y_preds)
    f, axes = plt.subplots(len(labels)//4, 4, figsize=(12, 16))
    axes = axes.ravel()
    for i in range(len(labels)):
        disp = ConfusionMatrixDisplay(cm[i], display_labels=["N", "P"])
        disp.plot(ax=axes[i], values_format='.0f')
        disp.ax_.set_title(f'{labels[i]}')
        if i<4*(len(labels)//4-1):
            disp.ax_.set_xlabel('')
        if i%4!=0:
            disp.ax_.set_ylabel('')
        disp.im_.colorbar.remove()

    f.tight_layout()
    plt.subplots_adjust(wspace=0)
    f.colorbar(disp.im_, ax=axes)
    plt.savefig(filepath)

    

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument("--taxonomy", type=str, default="original", help="Taxonomy (original, ekman, group, all)")
    cli_parser.add_argument("--ncomponents", type=int, default=6, help="Number of Clusters (2-27)")
    cli_parser.add_argument("--get_tsne", type=bool, default=False, help="Whether to save a TSNE plot.")
    cli_parser.add_argument("--get_confusion", type=bool, default=False, help="Whether to save a confusion matrix plot.")
    cli_args = cli_parser.parse_args()
    args = get_args(cli_args)
    output_dir = os.path.join(args.output_dir, "visualization")
    init_logger()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    label_list = get_labels(args)
    logger.info(f"Number of classes: {len(label_list)}")
    model = load_model(args, label_list)
    classifier_weights = visualize_last_weight(args, model)
    if args.get_tsne:
        logger.info("Get a TSNE plot of the classifier layer.")
        get_tsn_plot(classifier_weights, label_list, os.path.join(output_dir, "layer_embed_tsn.png"))
        logger.info("DONE!")
    if args.get_confusion:
        logger.info("Get confusion matrices plots.")
        best_dir = os.path.join(args.output_dir, "current_best")
        tokenizer_dir = os.path.join(best_dir, "tokenizer")
        args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        model.to(args.device)
        modes = ["train", "dev", "test"]
        for mode in modes:
            logger.info(f"Load {mode}-data.")
            dataloader = get_dataloader(args, mode, tokenizer_dir)
            logger.info(f"Get prediction and true.")
            y_preds, y_true = get_pred_and_true(args, dataloader, model)
            logger.info(f"Shape of predictions: {y_preds.shape}")
            logger.info(f"Shape of true label ids: {y_true.shape}")
            logger.info(f"Plot confusion matrices.")
            plot_confusion_matrix(y_preds, y_true, label_list, mode, os.path.join(output_dir,f"{mode}-confusion.png"))
            get_confusion_table(y_preds, y_true, label_list, mode, os.path.join(output_dir,f"{mode}-result.csv"))
            logger.info("DONE!")        

    dot_sim = classifier_weights @ classifier_weights.transpose()
    # Dot product similarities between pairs of labels
    dot_sim_df = pd.DataFrame(dot_sim, columns=label_list, index=label_list)
    n_components = cli_args.ncomponents
    scs = SpectralClustering(n_components).fit(dot_sim_df)
    for c in range(n_components):
        print([label for i, label in enumerate(label_list) if scs.labels_[i] == c])