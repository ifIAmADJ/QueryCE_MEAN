import os
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset

from lab.models import *
from utils.data_load import data_context as dc
from typing import Union


class TrainingQueryFeatsSet(torch.utils.data.Dataset):

    def __init__(self, wrapped_feats, cards):
        assert len(wrapped_feats) == len(cards), "feats and labels should have same length."
        self.x = wrapped_feats
        self.y = cards

    def __getitem__(self, item):
        wrapped_tensor_feats = tuple(torch.tensor(row, dtype=torch.float32) for row in self.x[item])
        return wrapped_tensor_feats, torch.tensor(self.y[item], dtype=torch.float32)

    def __len__(self):
        return len(self.x)


def from_csv(csv_path: str, data_context: dc.AbstractSqlEncoder, take: Union[int, float] = 1.0):
    df = pd.read_csv(csv_path)
    assert (df.columns.values == ['sql', 'label']).all(), "data source must contain 'sql' and 'label'"
    dumped_name = csv_path.split("/")[-1].split(".csv")[0] + f"_take{take}_bin{data_context.bins}.pkl"
    if os.path.exists(f"cached/training_set/{dumped_name}"):
        try:
            with open(f"cached/training_set/{dumped_name}", mode="rb") as f:
                dumped = pickle.load(f)
                assert isinstance(dumped, TrainingQueryFeatsSet)
                return dumped
        except:
            print("fail to load dumped training set, reload and dumping it again")

    if 0 < take <= 1.0:
        df = df.head(int(take * len(df)))
    else:
        assert isinstance(take, int)
        df = df.tail(take)

    feats = df['sql'].map(lambda x: data_context.encode_sql(x))
    log_cards = np.log(df['label'])
    dataset = TrainingQueryFeatsSet(feats, log_cards)

    with open(f"cached/training_set/{dumped_name}", mode="wb+") as f:
        pickle.dump(dataset, f)
        print("for quickly load next time, this dataset will be saved to ./cached/training_set/.")
    return dataset