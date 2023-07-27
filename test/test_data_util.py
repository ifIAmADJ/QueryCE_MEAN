import unittest

import numpy as np

import utils.data_load as dl


class TestDataUtil(unittest.TestCase):

    def test_parse_sql_within_context(self):
        ds = dl.parse_multi_csv_as_db(
            db_name="imdb",
            bins=128,
            dumping_path="../schema_meta"
        )

        imdb = dl.DBContextBase(ds)
        movie_info = imdb.db_schema.table_cells['movie_info']
        hm = movie_info.table_heatmap
        read_hist = movie_info.real_hist

        np.set_printoptions(suppress=True)
        print(hm)
        print(read_hist)


    def test_apply_db_context(self):
        imdb = dl.apply_db_context("imdb", bins=128, dumping_path="../schema_meta")

        # A sql from IMDB workload.
        sql = "SELECT COUNT(*) FROM title t,movie_companies mc,cast_info ci WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND mc.company_id<233881 AND mc.company_type_id=1 AND ci.person_id<190941 AND ci.role_id=10"

        np.set_printoptions(threshold=np.inf, suppress=True)
        query = imdb.parse_within_context(sql)
        print(query)

    def test_imdb_read(self):
        ds = dl.parse_multi_csv_as_db(
            db_name="imdb",
            bins=128,
            dumping_path="../schema_meta"
        )

        imdb = dl.DBContextBase(ds)

        for cell in imdb.db_schema.table_cells:
            c = imdb.db_schema.table_cells[cell]
            print(c.category_attrs)
