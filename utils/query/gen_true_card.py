import inspect
import logging
import math
import os
import random
import time
from enum import Enum
from typing import List, Tuple

import numpy as np
import pandas as pd
import tqdm

import utils.data_load.db_schema as sc
from utils.data_load.table_cell import TableMeta
from utils.query.db import MySQL, DB


class DataType(Enum):
    INT = 0,
    FLOAT = 1


def gen_sql_from_DBSchema(db_schema: sc.DBSchema, data_type: DataType = DataType.INT, ignore_id_col=True):
    assert isinstance(db_schema, sc.DBSchema), "not a right type. load datasets and convert them as a DBSchema."

    pick = random.choice
    tables = db_schema.tables

    table2idx: dict = {}
    idx = 0
    for table in tables:
        table2idx[table] = idx
        idx += 1
    idx2table = dict(zip(table2idx.values(), table2idx.keys()))

    join_graph = [[0 for _ in range(len(tables))] for _ in range(len(tables))]
    join_matrix_mapping = [[() for _ in range(len(tables))] for _ in range(len(tables))]

    for join_pair in db_schema.joins:
        ((t1, _), (t2, _)) = join_pair
        join_graph[table2idx[t1]][table2idx[t2]] = 1
        join_graph[table2idx[t2]][table2idx[t1]] = 1

        join_matrix_mapping[table2idx[t1]][table2idx[t2]] = join_pair
        join_matrix_mapping[table2idx[t2]][table2idx[t1]] = join_pair

    join_graph = np.array(join_graph, dtype=int)

    def sql_gen(max_tables_num: int, max_predicates_num: int) -> str:

        def zeros():
            return np.zeros(len(tables), dtype=int)

        def ones():
            return np.ones(len(tables), dtype=int)

        real_max_tables_num = min(len(tables), max_tables_num)
        real_tables_num = pick(range(1, real_max_tables_num + 1))

        # [1/3] pick available join pairs
        roamed_table_idxes = zeros()
        unseen_table_idxes = ones()  # unseen == ~roamed_table_idxes
        acc = zeros()
        reachable = zeros()
        picked_joins = set()

        start = pick(range(len(tables)))
        logging.debug(f"choose table: {idx2table[start]}(idx: {start})， roam {real_tables_num} tables")
        roamed_table_idxes[start] = 1
        unseen_table_idxes[start] = 0

        for _ in range(real_tables_num - 1):

            [idxes] = np.where(roamed_table_idxes == 1)
            for i in idxes:
                acc = acc | join_graph[i]

            for i in range(len(tables)):
                if roamed_table_idxes[i] == 0 and acc[i] == 1:
                    reachable[i] = 1
            reachable = reachable & unseen_table_idxes

            # (v1, v2) as a edge
            [candidate_v1] = np.where(reachable == 1)
            picked_v1 = pick(candidate_v1)
            [candidate_v2] = np.where(join_graph[picked_v1] & roamed_table_idxes == 1)
            picked_v2 = pick(candidate_v2)

            roamed_table_idxes[picked_v1] = 1
            unseen_table_idxes[picked_v1] = 0

            picked_join = join_matrix_mapping[picked_v1][picked_v2]
            assert picked_join != (), "can not find this join pair."
            picked_joins.add(picked_join)

        logging.debug(f"picked joins = {picked_joins}")

        # [2/3] extract tables from picked join pairs
        picked_tables = []
        for table_idx, signal in enumerate(roamed_table_idxes):
            if signal == 1:
                picked_tables += [idx2table[table_idx]]

        logging.debug(f"picked tables = {picked_tables}")

        # [3/3] generate predicate(s) for synthetic sql, at least 1.
        candidate_cols_meta_info = pd.DataFrame(data=[], columns=TableMeta.df_columns())
        real_predicate_num = pick(range(1, max_predicates_num))
        logging.debug(f"generate {real_predicate_num} predicates.")

        for t in picked_tables:
            table_cell = db_schema.table_cells[t]
            candidate_cols_meta_info = pd.concat([candidate_cols_meta_info, table_cell.table_meta], axis=0)

        if ignore_id_col:
            candidate_cols_meta_info = candidate_cols_meta_info.query(f"{TableMeta.COLUMN} != 'id'")

        picked_cols = candidate_cols_meta_info.sample(real_predicate_num)
        synthetic_predicates = []

        for col in picked_cols.values:
            # [0]: table, [1]: column, [2]: min_val, [3]: max_val, [4]: unique_count (not used)
            [t_name, c_name, col_min_val, col_max_val, _] = col

            single_predicate = ""
            pivot = np.round(random.uniform(col_min_val, col_max_val), 2)
            if data_type is DataType.INT:
                pivot = pivot.astype(int)
            else:
                pivot = np.around(pivot, decimals=2)

            predicate_types = ['point', 'range']
            predicate_type = pick(predicate_types)
            if predicate_type == 'point':
                op = [">", "=", "<"]
                single_predicate = f"{t_name}.{c_name}{pick(op)}{pivot}"
            elif predicate_type == 'range':
                width = random.uniform(0, col_max_val - col_min_val)
                lb = max(col_min_val, pivot - (width / 2))
                ub = min(col_max_val, pivot + (width / 2))

                if data_type is DataType.INT:
                    lb, ub = int(lb), int(ub)
                else:
                    lb = np.around(lb, decimals=2)
                    ub = np.around(ub, decimals=2)
                single_predicate = f"{t_name}.{c_name}>{lb} AND {t_name}.{c_name}<{ub}"

            synthetic_predicates += [single_predicate]

        joins_clause = " AND ".join([f"{tb1}.{k1}={tb2}.{k2}" for ((tb1, k1), (tb2, k2)) in picked_joins])
        joins_clause = joins_clause + " AND " if joins_clause else ""
        tables_clause = ",".join(picked_tables)
        predicates_clause = " AND ".join(synthetic_predicates)

        logging.debug(f"table-clause: {tables_clause}")
        logging.debug(f"join-clause: {joins_clause if joins_clause else None}")
        logging.debug(f"predicate-clause: {predicates_clause}")

        sql = f"SELECT COUNT(*) FROM {tables_clause} WHERE {joins_clause}{predicates_clause}"

        yield sql
        yield from sql_gen(max_tables_num, max_predicates_num)

    return sql_gen


def gen_true_label_from_db(db: DB, sql_gen, max_tables_num, max_predicates_num, gen_num: int, output_path: str) -> List[
    Tuple[str, int]]:
    logging.basicConfig(level=logging.DEBUG)

    assert inspect.isgeneratorfunction(sql_gen), "2nd param: sql_gen, must be a generator function"

    if not os.path.exists(output_path):
        df = pd.DataFrame(data=[], columns=['sql', 'label'])
        df.to_csv(path_or_buf=output_path, mode="w", index=False, header=True)
        logging.debug(f"create new file: {output_path} and append.")

    query_gen = sql_gen
    cache = []

    cache_size = max(int(gen_num / 100), 5)
    batch_num = int(math.ceil(gen_num / cache_size))
    batch_count = 0
    logging.debug(f"cache size = {cache_size}, batch_num = {batch_num}")
    count = 0

    with tqdm.tqdm(total=gen_num) as pbar:
        while count < gen_num:
            sql = next(query_gen(max_tables_num=max_tables_num, max_predicates_num=max_predicates_num))
            logging.info(f"gen sql: {sql}")
            label = db.true_card_of(sql)
            if label != 0:
                cache += [(sql, label)]
                count += 1

                if len(cache) == cache_size or count >= gen_num:
                    batch_count += 1
                    logging.info(f"[{batch_count}/{batch_num}] flush >> {output_path} and clear the cache.")
                    batch = pd.DataFrame(data=cache)
                    batch.to_csv(path_or_buf=output_path, mode="a", index=False, header=False)
                    cache.clear()

                pbar.update(1)
                logging.info(f"sql card = {label}")

    return cache


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # show processlist;
    # kill N;
    imdb = sc.parse_multi_csv_as_db("imdb", bins=64, dumping_path="../../schema_meta")
    imdb_sql_gen = gen_sql_from_DBSchema(db_schema=imdb, ignore_id_col=True)
    db = MySQL("imdb")

    time1 = time.time()
    gen_true_label_from_db(db, imdb_sql_gen, max_tables_num=3, max_predicates_num=3, gen_num=100,
                           output_path="../../data/chunks_imdb/m1")
    time2 = (time.time() - time1) * 1000

    print(f"take time: {time2}ms")

