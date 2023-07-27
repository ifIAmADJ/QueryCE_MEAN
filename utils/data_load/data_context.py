import collections
import itertools
import math
from abc import ABC, abstractmethod
from typing import List, Optional, Dict

import numpy as np
from pandas import DataFrame

import utils.data_load.config as cfg
from utils.data_load.db_schema import DBSchema, parse_multi_csv_as_db, TableMeta
from utils.query.sql_2_query import Bound, parse, Query


def normalize(min_val, max_val, decimals=3):
    def f(value):
        result = np.around((value - min_val) / (max_val - min_val), decimals=decimals)
        result = np.maximum(0, result)
        result = np.minimum(1, result)
        assert 0 <= result <= 1, f"min_val = {min_val}, max_val = {max_val}, but value = {value}"
        return result

    return f


def fill_up_range(table_meta: DataFrame, query: Query, col_name: str, table_name: str) -> None:
    (lb, ub) = query.ranges[(col_name, table_name)]
    stat = table_meta.query(f'column == "{col_name}" and table == "{table_name}"')
    min_val = stat["min_val"].array[0]
    max_val = stat["max_val"].array[0]

    if lb is Bound.Lower:
        query.ranges[(col_name, table_name)] = (min_val, ub)
    if ub is Bound.Upper:
        query.ranges[(col_name, table_name)] = (lb, max_val)


class AbstractSqlEncoder(ABC):
    """
        A AbstractSqlEncoder should convert a string sql to a set of feature vector(s) by ```encode_sql``` method,
        depends on how we encode sql.
    """

    def __init__(self):
        self.bins = 0

    @abstractmethod
    def encode_sql(self, sql: str) -> List: pass


class DBContextBase(AbstractSqlEncoder):

    def __init__(self, sc: DBSchema):

        super().__init__()
        self.db_schema: DBSchema = sc
        self.bins = sc.bins
        self.col2indices = self._init_col2indices()
        self.table2indices = self._init_table2indices()
        self.join2indices = self._init_join2indices()

        cols = len(self.col2indices)
        self.proj_feat_shape = (cols, self.bins)
        self.ranges_feat_len = 2 * cols
        self.tables_feat_len = len(self.table2indices)
        self.joins_feat_len = len(self.join2indices)
        self.indicator_feat_len = cols

    def parse_within_context(self, sql: str) -> Query:

        table_cells = self.db_schema.table_cells
        parsed = parse(sql)
        for col, table in parsed.ranges:
            t_meta: DataFrame = table_cells[table].table_meta
            fill_up_range(t_meta, parsed, col, table)

        return parsed

    def encode_sql(self, sql: str) -> List:
        query = self.parse_within_context(sql)
        return [self.ranges_feat(query), self.projs_feat(query), self.tables_feat(query), self.joins_feat(query)]

    def _init_table2indices(self):

        tables = self.db_schema.tables
        idx: int = 0
        tables_idx_dict: dict = {}
        for table in tables:
            tables_idx_dict[table] = idx
            idx += 1

        return tables_idx_dict

    def _init_join2indices(self):

        joins = self.db_schema.joins
        idx: int = 0
        joins_idx_dict = collections.OrderedDict()
        for join_pair in joins:
            joins_idx_dict[join_pair] = idx
            idx += 1

        return joins_idx_dict

    def _init_col2indices(self):

        tables = self.db_schema.tables
        idx = 0
        col2indices: dict = {}
        for table in tables:
            table_cell = self.db_schema.table_cells[table]
            cols = table_cell.table_meta[TableMeta.COLUMN]

            for col in cols:
                col2indices[(col, table)] = idx
                idx += 1

        return col2indices

    # we use one-hot as base encoding method.
    def tables_feat(self, query: Query):

        feat_len = len(self.table2indices)
        table_f = [0 for _ in range(feat_len)]
        for table in query.tables:
            table_f[self.table2indices[table]] = 1

        return table_f

    # one-hot encoding
    def joins_feat(self, query: Query):

        feat_len = len(self.join2indices)
        joins_f = [0 for _ in range(feat_len)]
        for join in query.joins:

            # "A.a1 = B.b1" is same as "B.b1 = A.a1"
            (p1, p2) = join
            if (p1, p2) in self.join2indices:
                joins_f[self.join2indices[(p1, p2)]] = 1
            elif (p2, p1) in self.join2indices:
                joins_f[self.join2indices[(p2, p1)]] = 1
            else:
                raise Exception(f"can not find the index of join pair: {p1} <-> {p2}")

        return joins_f

    # one-hot encoding
    def ranges_feat(self, query: Query):

        feat_len = len(self.col2indices) * 2
        # normalized, so lb_i = 0, ub_i = 1.
        range_f = [0. if x % 2 == 0 else 1. for x in range(feat_len)]

        for each_range in query.ranges:
            col, table = each_range
            table_cell = self.db_schema.table_cells[table]

            col_info = table_cell.table_meta.query(f"{TableMeta.COLUMN} == '{col}'").iloc[0]
            col_min_val = col_info[TableMeta.MIN_VAL]
            col_max_val = col_info[TableMeta.MAX_VAL]

            lb, ub = query.ranges[each_range]

            lb = min(lb, col_max_val)
            ub = max(ub, col_min_val)
            norm = normalize(col_min_val, col_max_val)
            lb_normed = norm(lb)
            ub_normed = norm(ub)

            index = self.col2indices[each_range]
            range_f[index * 2] = lb_normed
            range_f[index * 2 + 1] = ub_normed

        return range_f

    def projs_feat(self, query: Query):

        col_num = len(self.col2indices)
        bin_num = self.bins
        shape = (col_num, bin_num)  # cols × bins
        projs_f = np.zeros(shape)

        for each_range in query.ranges:
            (col, table) = each_range
            table_cell = self.db_schema.table_cells[table]

            hotspot_idx = table_cell.table_meta.query(f"{TableMeta.COLUMN} == '{col}'").index[0]
            slice = table_cell.table_heatmap[hotspot_idx]

            col_info = table_cell.table_meta.query(f"{TableMeta.COLUMN} == '{col}'").iloc[0]
            col_min_val = col_info[TableMeta.MIN_VAL]
            col_max_val = col_info[TableMeta.MAX_VAL]

            r_chunk = (col_max_val - col_min_val) / bin_num
            lb, ub = query.ranges[each_range]

            lb_ = max(lb, col_min_val)
            ub_ = min(ub, col_max_val)

            lb_idx = int(math.floor((lb_ - col_min_val) / r_chunk))
            ub_idx = int(math.floor((ub_ - col_min_val) / r_chunk))

            proj_f_idx = self.col2indices[(col, table)]

            proj_slice = np.zeros(bin_num)
            proj_slice[lb_idx:ub_idx] = slice[lb_idx:ub_idx]
            projs_f[proj_f_idx] = proj_slice

        return projs_f


class DBContext(DBContextBase):

    def _filtered_samples(self, query: Query) -> Dict[str, DataFrame]:
        selected_samples = dict()
        selected_predicates = query.predicates
        selected_tables = query.tables

        for table in selected_tables:
            table_cell = self.db_schema.table_cells[table]
            selected_samples[table] = table_cell.t_sample.copy()

        preds_group = itertools.groupby(selected_predicates, lambda col_table: col_table[1])

        for table, group in preds_group:
            grp = list(group)
            samples = selected_samples[table]
            for c, t in grp:
                assert t == table
                ops = selected_predicates[(c, t)]
                for op, val in ops:
                    op = "==" if op == "=" else op  # convert SQL's "=" as Python's "==".
                    samples = samples.query(f"{c} {op} {val}")

                selected_samples[t] = samples

        return selected_samples

    def joins_feat(self, query: Query, filtered_samples: Dict[str, DataFrame] = None):

        feat_len = len(self.join2indices)
        joins_f = [0 for _ in range(2 * feat_len)]

        pre_set_min_val = 1
        pre_set_max_val = 2_530_000
        pre_set_bins = 64

        for join in query.joins:

            # "A.a1 = B.b1" is same as "B.b1 = A.a1"
            (p1, p2) = join

            (c1, t1), (c2, t2) = p1, p2
            filtered_samples = self._filtered_samples(query)

            sample1_keys = filtered_samples[t1][c1]
            sample2_keys = filtered_samples[t2][c2]

            [hist1, _] = np.histogram(sample1_keys, range=(pre_set_min_val, pre_set_max_val), bins=pre_set_bins)
            [hist2, _] = np.histogram(sample2_keys, range=(pre_set_min_val, pre_set_max_val), bins=pre_set_bins)

            h = hist1 * hist2

            join_eval = max(h.sum(axis=0), 1)
            join_eval = max(np.log(join_eval), 0.001)

            if (p1, p2) in self.join2indices:
                joins_f[self.join2indices[(p1, p2)] * 2] = 1
                joins_f[self.join2indices[(p1, p2)] * 2 + 1] = join_eval
            elif (p2, p1) in self.join2indices:
                joins_f[self.join2indices[(p2, p1)] * 2] = 1
                joins_f[self.join2indices[(p2, p1)] * 2 + 1] = join_eval
            else:
                raise Exception(f"can not find the index of join pair: {p1} <-> {p2}")

        return joins_f


    def space_feat(self, query: Query, bin_num=128):

        space_f = [0., 0., 0.] * len(self.col2indices)
        for each_range in query.ranges:
            (col, table) = each_range
            table_cell = self.db_schema.table_cells[table]

            col_info = table_cell.table_meta.query(f"{TableMeta.COLUMN} == '{col}'").iloc[0]
            col_min_val = col_info[TableMeta.MIN_VAL]
            col_max_val = col_info[TableMeta.MAX_VAL]
            category = col_info[TableMeta.CATEGORY]

            lb, ub = query.ranges[each_range]
            lb_ = max(lb, col_min_val)
            ub_ = min(ub, col_max_val)
            proj_f_idx = self.col2indices[(col, table)]
            normal = normalize(col_min_val, col_max_val)

            if not category:
                r_chunk = (col_max_val - col_min_val) / bin_num
                hotspot_idx = table_cell.table_meta.query(f"{TableMeta.COLUMN} == '{col}'").index[0]
                slice = table_cell.real_hist[hotspot_idx]
                lb_idx = int(math.floor((lb_ - col_min_val) / r_chunk))
                ub_idx = int(math.floor((ub_ - col_min_val) / r_chunk))
                density = max(sum(slice[lb_idx:ub_idx]), 1)
            else:
                count = 1
                for k in table_cell.category_attrs[col]:
                    count += table_cell.category_attrs[col][k] if lb_ <= k <= ub_ else 0
                density = count

            space_f[3 * proj_f_idx] = normal(lb_)
            space_f[3 * proj_f_idx + 1] = normal(ub_)
            space_f[3 * proj_f_idx + 2] = np.around(np.log(density), 3)

            # print(f"in the {col}, {table}, range_from {lb_},{ub_}, normed {normal(lb_)}, {normal(ub_)}. max + {col_max_val}, min = {col_min_val}  density = {density}, is_cate = {category}")

        return space_f

    def ranges_feat(self, query: Query):

        # [lb, ub, =]
        range_f = [0., 0., 0] * len(self.col2indices)

        for each_range in query.ranges:
            col, table = each_range
            table_cell = self.db_schema.table_cells[table]

            col_info = table_cell.table_meta.query(f"{TableMeta.COLUMN} == '{col}'").iloc[0]
            col_min_val = col_info[TableMeta.MIN_VAL]
            col_max_val = col_info[TableMeta.MAX_VAL]

            lb, ub = query.ranges[each_range]

            lb = min(lb, col_max_val)
            ub = max(ub, col_min_val)
            norm = normalize(col_min_val, col_max_val)
            lb_normed = norm(lb)
            ub_normed = norm(ub)

            index = self.col2indices[each_range]
            range_f[index * 3] = lb_normed
            range_f[index * 3 + 1] = ub_normed

        for (col, t) in query.predicates:
            p_list = query.predicates[(col, t)]
            start_idx = self.col2indices[(col, t)] * 3
            for (op, _) in p_list:
                op_idx = start_idx + 2
                if op == '=':
                    range_f[op_idx] = 1

        return range_f

    def tables_feat(self, query: Query, filtered_samples: Dict[str, DataFrame] = None, samples_size: int = 1000):

        table_f = np.zeros(shape=2 * len(self.table2indices))

        for table in query.tables:
            t_sample = filtered_samples[table]
            sel = len(t_sample) / samples_size
            est_card = max(sel * self.db_schema.table_cells[table].table_card, 1)
            log_card = max(np.log(est_card), 0.001)

            table_f[self.table2indices[table] * 2] = 1
            table_f[self.table2indices[table] * 2 + 1] = log_card

        return table_f

    @staticmethod
    def small_scale_hint(ranges_f, tables_f, joins_f) -> List:

        ranges_h = ranges_f[2::3]
        tables_h = [1 if [1, 0.001] == [a, b] else 0 for a, b in zip(tables_f[::2], tables_f[1::2])]
        joins_h = [1 if [1, 0.001] == [a, b] else 0 for a, b in zip(tables_f[::2], joins_f[1::2])]

        range_s = 1 if any(e == 1 for e in ranges_h) else 0
        tables_s = 1 if any(e == 1 for e in tables_h) else 0
        joins_s = 1 if any(e == 1 for e in joins_h) else 0

        return [range_s, tables_s, joins_s]

    def encode_sql(self, sql: str) -> List:

        query = self.parse_within_context(sql)
        filtered_samples = self._filtered_samples(query)  # only-read, share safely.

        ranges_feat = self.ranges_feat(query)
        hist_projection_feat = self.projs_feat(query)
        tables_feat = self.tables_feat(query, filtered_samples)
        joins_feat = self.joins_feat(query, filtered_samples)

        # space_feat = self.space_feat(query)
        # tables_feat = super().tables_feat(query)
        # joins_feat = super().joins_feat(query)

        small_scale_vote_feat = self.small_scale_hint(ranges_feat, tables_feat, joins_feat)
        table_mask = super().tables_feat(query)
        return [ranges_feat, hist_projection_feat, tables_feat, joins_feat, small_scale_vote_feat, table_mask]


def apply_db_context(db_name: str, bins: int, dumping_path: Optional[str],
                     csv_root_path: str = cfg.DB_ROOT_PATH) -> DBContext:
    sc = parse_multi_csv_as_db(db_name=db_name, bins=bins, dumping_path=dumping_path, root_path=csv_root_path)
    return DBContext(sc)
