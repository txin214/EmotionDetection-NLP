import os
import pandas as pd

data_dir = "data"
levels = ["original", "ekman", "group"]
files = ["train.tsv", "dev.tsv", "test.tsv"]
df_level_dict = {}
df_split_dict = {}
for file in files:
    for level in levels:
        file_to_read = os.path.join(level, file)
        file_path = os.path.join(data_dir, file_to_read)
        df_level_dict[level] = pd.read_csv(file_path, sep="\t", 
                            header=None, names=["text", f"label-{level}", "user_id"])
    split = file.replace(".tsv", "")
    df_split_dict[split] = pd.merge(df_level_dict["original"], df_level_dict["ekman"], how="left", on=["text", "user_id"])
    df_split_dict[split] = pd.merge(df_split_dict[split], df_level_dict["group"], how="left", on=["text", "user_id"])
    df_split_dict[split].to_csv(f"data/all/{file}", sep="\t")
