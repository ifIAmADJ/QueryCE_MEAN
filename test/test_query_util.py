import unittest

from utils.query.db import MySQL
import utils.query.gen_true_card as gc
import utils.data_load as dl
from utils.query.sql_2_query import parse


class TestQueryUtil(unittest.TestCase):

    def test_for_pure_sql_parse(self):
        """
            convert a sql string to a instance of Query,
        """

        query = parse(
            "SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.production_year>1980 AND t.production_year<1995;"
        )

        print(query)

    def test_gen_label_from_mysql(self):
        """
            get ground truth card val of a query, All db object extends DB abstract class.
        """
        db = MySQL("pandas")
        test_sql = "SELECT COUNT(*) FROM forest WHERE forest.Elevation > 2000;"
        res = db.true_card_of(test_sql)
        print(res)

    def test_gen_sql_from_db(self):

        db = MySQL("imdb")
        imdb = dl.parse_multi_csv_as_db("imdb", bins=64, dumping_path="../schema_meta")
        imdb_sql_gen = gc.gen_sql_from_DBSchema(imdb)

        for _ in range(3):
            sql = next(imdb_sql_gen(max_tables_num=3, max_predicates_num=3))
            card = db.true_card_of(sql)
            print(f"generate sql: {sql}, card: {card}")


    def test_gen_sql_from_table(self):

        db = MySQL("pandas")
        forest = dl.parse_single_csv_as_table("forest", csv_path="C:\\Users\\liJunhu\\Desktop\\data\\forest10\\original.csv",bins=64)
        forest = dl.lift(forest, dumping_path="../schema_meta")  # TableCell -> DBSchema

        forest_sql_gen = gc.gen_sql_from_DBSchema(forest)
        for _ in range(3):
            sql = next(forest_sql_gen(max_tables_num=1, max_predicates_num=3))
            card = db.true_card_of(sql)
            print(f"generate sql: {sql}, and card: {card}")

