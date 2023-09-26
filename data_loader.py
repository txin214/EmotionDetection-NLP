import os
import logging
import torch
import pandas as pd
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, label=None, label_original=None, label_ekman=None, label_senti=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label
        self.label_original=label_original
        self.label_ekman = label_ekman
        self.label_senti = label_senti

def convert_to_one_hot_label(label, num_labels):
    one_hot_label = [0] * num_labels
    for l in label:
        one_hot_label[int(l)] = 1
    return one_hot_label

def convert_df_to_features(args, df, tokenizer, max_length):
    processor = GoEmotionsProcessor(args)
    num_labels = len(processor.get_labels())
    labels = df["label"].apply(lambda label: convert_to_one_hot_label(label.split(","), num_labels))
    '''
    The batch_encode_plus method is used to tokenize the sequences, 
    add special tokens (such as [CLS] and [SEP]) to mark the beginning and 
    end of each sequence, pad the sequences to a maximum length, 
    and truncate any sequences that are longer than maximum length tokens. 
    The resulting 
    token IDs, attention masks, and token type IDs (if applicable) 
    are returned as PyTorch tensors.
    '''
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

class GoEmotionsProcessor(object):
    """Processor for the GoEmotions data set """

    def __init__(self, args):
        self.args = args

    def get_labels(self):
        labels = []
        with open(os.path.join(self.args.data_dir, 
                               self.args.label_file), "r", encoding="utf-8") as f:
            for line in f:
                labels.append(line.rstrip())
        return labels

    def get_df(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None
        if mode == 'train':
            file_to_read = self.args.train_file
        elif mode == 'dev':
            file_to_read = self.args.dev_file
        elif mode == 'test':
            file_to_read = self.args.test_file
        else:
            raise ValueError("For mode, only train, dev, test is available")
        file_path = os.path.join(self.args.data_dir, file_to_read)
        df = pd.read_csv(file_path, sep="\t", 
                         header=None, names=["text", "label", "user_id"])
        if self.args.process is True:
            def process(x):
                x = x.split(",")
                if ('27' in x) and (len(x)>1):
                    x.remove('27')
                return ','.join(x)
            df["label"] = df['label'].apply(lambda x: process(x))
        
        return df


def load_and_cache_examples(args, tokenizer, mode):
    processor = GoEmotionsProcessor(args)
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            str(args.task),
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_len),
            mode
        )
    )
    if os.path.exists(cached_features_file) and args.load:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        df = processor.get_df(mode)

        features = convert_df_to_features(
            args, df, tokenizer, max_length=args.max_seq_len
        )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def load_all(args, tokenizer, mode):
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
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

        labels = []
        label_names = ["label-ekman", "label-group", "label-original"]
        labels = {}
        num_labels = {"label-ekman":7, "label-group":4, "label-original":28}
        for label_name in label_names:
            labels[label_name] = df[label_name].apply(lambda label: convert_to_one_hot_label(label.split(","), num_labels[label_name]))
        
        text_encoded = tokenizer.batch_encode_plus(
            df.text.values.tolist(),
            add_special_tokens=True,
            max_length=args.max_seq_len,
            padding='max_length',
            truncation=True
        )

        features = []
        for i in range(len(df)):
            inputs = {k: text_encoded[k][i] for k in text_encoded}
            feature = InputFeatures(**inputs, 
                                    label_original=labels["label-original"][i], 
                                    label_ekman=labels["label-ekman"][i],
                                    label_senti=labels["label-group"][i])
            features.append(feature)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels_original = torch.tensor([f.label_original for f in features], dtype=torch.float)
    all_labels_ekman = torch.tensor([f.label_ekman for f in features], dtype=torch.float)
    all_labels_senti = torch.tensor([f.label_senti for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels_original, all_labels_ekman, all_labels_senti)
    return dataset
    