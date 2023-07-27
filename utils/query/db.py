import json
from abc import ABC, abstractmethod

import pymysql

from utils.query.config import conn


class DB(ABC):

    @abstractmethod
    def true_card_of(self, query: str) -> int: pass

    @abstractmethod
    def est_card_of(self, query: str) -> int: pass


class MySQL(DB):
    def __init__(self, db_name):
        mysql = conn["mysql"]

        host = mysql["host"]
        port = mysql["port"]
        user = mysql["user"]
        pwd = mysql["pwd"]
        db = db_name

        self._conn = pymysql.connect(host=host, port=port, user=user, password=pwd, database=db)
        self._cursor = self._conn.cursor()

    def true_card_of(self, query: str) -> int:
        self._cursor.execute(query)
        label_, = self._cursor.fetchone()
        return label_

    def est_card_of(self, query: str) -> int:
        self._cursor.execute(f"EXPLAIN format = json {query}")
        explain_json, = self._cursor.fetchone()
        explain_dict = json.loads(explain_json)["query_block"]
        card = int(explain_dict["table"]["rows_examined_per_scan"])
        sel = float(explain_dict["table"]["filtered"]) * 1e-2 # 0 ~ 100.0 (%)
        return int(card * sel)