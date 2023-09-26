import os
import logging
import torch
import pandas as pd
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, label=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

def get_df(args, mode):
    """
    Args:
        mode: train, dev, test
    """
    file_to_read = None
    if mode == 'train':
        file_to_read = args.train_file
    elif mode == 'dev':
        file_to_read = args.dev_file
    elif mode == 'test':
        file_to_read = args.test_file
    else:
        raise ValueError("For mode, only train, dev, test is available")
    file_path = os.path.join(args.data_dir, file_to_read)
    df = pd.read_csv(file_path, sep="\t")
    df["is_neutral"] = df["is_neutral"].apply(lambda x:int(x))
    df = df.rename(columns={"label": "origin-label", "is_neutral": "label"})
    return df

def convert_to_one_hot_label(label, num_labels):
    one_hot_label = [0] * num_labels
    for l in label:
        one_hot_label[int(l)] = 1
    return one_hot_label

def convert_df_to_features(args, df, tokenizer, max_length):
    labels = df["label"]
    text_encoded = tokenizer.batch_encode_plus(
        df.text.values.tolist(),
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True
    )

    features = []
    for i in range(len(df)):
        inputs = {k: text_encoded[k][i] for k in text_encoded}
        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)
    return features

def load_and_cache_examples(args, tokenizer, mode, get_label=False):

    if get_label:
        df = get_df(args, mode)
        df = df.rename(columns={"label": "flag", "origin-label": "label"})
        num_labels = 28
        df["label"] = df["label"].apply(lambda label: convert_to_one_hot_label(label.split(","), num_labels))
        features = convert_df_to_features(
            args, df, tokenizer, max_length=args.max_seq_len
        )

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
        return dataset

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}_neutral".format(
            str(args.task),
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_len),
            mode
        )
    )

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        df = get_df(args, mode)
    
        features = convert_df_to_features(
            args, df, tokenizer, max_length=args.max_seq_len
        )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([[f.label] for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset