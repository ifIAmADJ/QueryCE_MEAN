from typing import List
import numpy as np
import pandas as pd


# Enum-Like, table headers of table_meta.
class TableMeta:

    TABLE = "table"
    COLUMN = "column"
    MIN_VAL = "min_val"
    MAX_VAL = "max_val"
    UNIQUE_COUNT = "unique_count"
    CATEGORY = "category"

    @staticmethod
    def df_columns() -> List[str]:
        t = TableMeta
        return [t.TABLE, t.COLUMN, t.MIN_VAL, t.MAX_VAL, t.UNIQUE_COUNT, t.CATEGORY]


class TableCell:
    def __init__(self, table_name: str,
                 table_heatmap: np.ndarray,
                 table_meta: pd.DataFrame,
                 table_card: int,
                 bins: int,
                 category_attrs: dict,
                 samples: pd.DataFrame,
                 real_hist: np.ndarray
                 ):
        self.table_name: str = table_name
        self.table_heatmap: np.ndarray = table_heatmap

        self.table_meta: pd.DataFrame = table_meta
        self.table_card: int = table_card

        self.bins: int = bins
        self.category_attrs: dict = category_attrs
        self.t_sample: pd.DataFrame = samples

        self.real_hist: np.ndarray = real_hist

    # extract meta-info of table from TableCell as a tuple.
    def unapply(self):
        return self.table_name, self.table_heatmap, self.table_meta, self.table_card

    def query_col(self, col_name: str):
        return self.table_meta.query(f"{TableMeta.COLUMN} == '{col_name}'").iloc[0]

def parse_single_csv_as_table(table_name: str, csv_path: str, bins: int = 64, samples_num: int = 1000) -> TableCell:

    df = pd.read_csv(csv_path)
    table_heatmap = np.empty(shape=(0, bins))
    table_meta = []
    table_card = len(df)

    real_hist = np.empty(shape=(0, bins))

    # samples
    samples = df.sample(samples_num)

    def normalize(val, min_v, max_v):
        return np.around((val - min_v) / (max_v - min_v), 3)

    category_attrs: dict = {}

    for series in df:
        unique_count = df[series].unique().size
        category = False
        if unique_count <= 15:
            category = True
            values = {}
            grp = df.groupby(series)
            [values.update({v: len(c)}) for v, c in grp]
            category_attrs.update({series: values})

        # hist_vector[0]: count num of buckets.
        # hist_vector[1]: value region of buckets.
        hist_vector = np.histogram(round(df[series], 3), bins=bins)
        min_val = hist_vector[1][0]
        max_val = hist_vector[1][-1]
        min_height = min(hist_vector[0])
        max_height = max(hist_vector[0])

        normed_hist = [normalize(h, min_height, max_height) for h in hist_vector[0]]
        table_heatmap = np.concatenate([table_heatmap, [normed_hist]], axis=0)
        real_hist = np.concatenate([real_hist, [hist_vector[0]]], axis=0)
        table_meta += [[table_name, series, min_val, max_val, unique_count, category]]

    table_meta = pd.DataFrame(data=table_meta, columns=TableMeta.df_columns())

    return TableCell(table_name, table_heatmap, table_meta, table_card, bins, category_attrs, samples, real_hist)
