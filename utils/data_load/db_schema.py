import os.path
import pickle
from typing import List, Tuple, Dict, Optional

import utils.data_load.config as cfg
from utils.data_load.table_cell import parse_single_csv_as_table, TableMeta, TableCell


class DBSchema:
    def __init__(self, db_name: str, table_cells: Dict[str, TableCell],
                 table_joins: List[Tuple[Tuple, Tuple]], bins: int):
        self.db_name: str = db_name
        self.table_cells: Dict[str, TableCell] = table_cells
        self.joins: List[Tuple[Tuple, Tuple]] = table_joins
        self.bins = bins
        self.tables: List[str] = [t for t in table_cells]

    def __repr__(self):
        joins_pair = []
        for zip_t in self.joins:
            ((k1, t1), (k2, t2)) = zip_t
            joins_pair += [f"'{t1}.{k1}={t2}.{k2}'"]

        return f"""
            db name: {self.db_name}
            As for now, we choose the 'one-hot' encoding method.
            Tables: {self.tables}
            Join key pairs: {joins_pair}
            Bin num: {self.bins}
        """


# lift a TableCell to SchemaContext.
def lift(table_cell: TableCell, dumping_path: str) -> DBSchema:
    db_name = table_cell.table_name
    maybe_sc = load_db_schema_from_pkl(db_name=db_name, bins=table_cell.bins, dumping_path=dumping_path)
    if maybe_sc is None:
        print(f"lift single table: {db_name} as a DB.")
        t_name = table_cell.table_name
        sc = DBSchema(db_name=t_name, table_cells={t_name: table_cell}, table_joins=list(), bins=table_cell.bins)
        save_schema_ctx_as_pkl(sc, dumping_path)
        return sc
    else:
        print(f"{db_name}_{table_cell.bins}bin.pkl has been cached previously, load it from disk.")
        return maybe_sc


def save_schema_ctx_as_pkl(sc: DBSchema, dumping_path: str) -> None:
    file_name = f"{sc.db_name}_{sc.bins}bin.pkl"
    with open(f"{dumping_path}/{file_name}", mode="wb+") as f:
        pickle.dump(sc, f)
        print(f"saved to local path: {dumping_path}/{file_name}")


def load_db_schema_from_pkl(db_name: str, bins: int, dumping_path) -> Optional[DBSchema]:
    file_name = f"{db_name}_{bins}bin.pkl"
    if not os.path.exists(f"{dumping_path}/{file_name}"):
        return None

    with open(f"{dumping_path}/{file_name}", mode="rb") as f:
        schema: DBSchema = pickle.load(f)
        return schema


def find_matched_csv(root_path: str, tables_and_unpick_cols: Dict):
    files = os.listdir(root_path)
    target_csv_files = list(filter(lambda file: file.endswith(".csv"), files))
    target_csv_files = list(
        filter(lambda picked_t: picked_t.removesuffix(".csv") in tables_and_unpick_cols, target_csv_files))

    def lack_of(src_tables, wanted_pick_table_and_cols):
        lack = []
        for table in wanted_pick_table_and_cols:
            if table + ".csv" not in src_tables:
                lack += [table]

        return ", ".join(lack)

    assert len(target_csv_files) == len(tables_and_unpick_cols), \
        f"lack of table(s): {lack_of(target_csv_files, tables_and_unpick_cols)}"

    return target_csv_files


def parse_multi_csv_as_db(db_name: str, bins: int = 64,
                          dumping_path: Optional[str] = None,
                          root_path: str = cfg.DB_ROOT_PATH,
                          tables_and_unpick_cols=cfg.tables_and_unpick_cols,
                          join_pairs: List = cfg.join_pairs
                          ) -> DBSchema:

    if dumping_path is None:
        maybe_sc = None
    else:
        maybe_sc = load_db_schema_from_pkl(db_name=db_name, bins=bins, dumping_path=dumping_path)

    if maybe_sc is None:
        print(f"read data from data root path: {root_path}")

    if isinstance(maybe_sc, DBSchema):
        print(f"{db_name}_{bins}bin.pkl has been cached previously, load it from disk.")
        return maybe_sc

    csv_file_prefix = ".csv"
    csv_files = find_matched_csv(root_path=root_path, tables_and_unpick_cols=tables_and_unpick_cols)
    for csv_file in csv_files:
        from_path = f"{cfg.DB_ROOT_PATH}/{csv_file}"
        assert os.path.exists(from_path), f"path: {from_path} is not exist."

    table_cells: dict = {}
    for csv_file in csv_files:
        table: str = csv_file.split(csv_file_prefix)[0]

        from_path = f"{cfg.DB_ROOT_PATH}/{csv_file}"
        table_cell = parse_single_csv_as_table(table, from_path, bins)
        table_cells[table] = table_cell

    table_names = [x for x in table_cells]

    # check join-pairs
    assert len(set(join_pairs)) == len(join_pairs), "some join pairs are duplicated, please check your config."
    for join_pair in join_pairs:
        ((key1, table1), (key2, table2)) = join_pair
        assert table1 in table_names, f"table: {table1} is not in db: {db_name}"
        assert table2 in table_names, f"table: {table2} is not in db: {db_name}"

        t1_meta = table_cells[table1].table_meta
        assert key1 in t1_meta[
            TableMeta.COLUMN].values, \
            f"column: {key1} is not in table: {table1}, columns = {t1_meta[TableMeta.COLUMN]} "

        t2_meta = table_cells[table2].table_meta
        assert key2 in t2_meta[TableMeta.COLUMN].values, \
            f"column: {key2} is not in table: {table2}, columns = {t2_meta[TableMeta.COLUMN]}"

    schema = DBSchema(db_name=db_name, table_cells=table_cells, table_joins=join_pairs, bins=bins)

    if dumping_path is not None:
        save_schema_ctx_as_pkl(sc=schema, dumping_path=dumping_path)

    return schema
