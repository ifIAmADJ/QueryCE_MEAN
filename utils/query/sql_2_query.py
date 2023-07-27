from typing import Dict, Tuple, List, Union, Set, Any

import sqlparse as sp
import sqlparse.sql
from enum import Enum


class Bound(Enum):
    Lower = 0  # e.g: 'Std.age < 18' ->  ('age','Std') : (Bound.Lower, 18))
    Upper = 1  # e.g: 'age > 24'     ->  ('age','Std') : (24, Bound.Upper))


class Query:
    def __init__(self, tables: list, joins: Set[Tuple[Tuple[str, str], Tuple[str, str]]],
                 ranges: Dict[Tuple[str, str], Tuple[Any, Any]], predicates: Dict[Tuple[str, str], List[Tuple[str, float]]]):
        self.tables: list = tables
        self.joins: Set[Tuple[Tuple, Tuple]] = joins
        self.ranges: Dict[Tuple[str, str], Tuple[Any, Any]] = ranges
        self.predicates: Dict[Tuple[str, str], List[Tuple[str, float]]] = predicates

    def __repr__(self):
        return f""" 
            tables : {self.tables},
            joins : {self.joins},
            ranges : {self.ranges}
            predicates: {self.predicates}
        """


def parse(query: str) -> Query:
    def get_name_and_alias(sub_stmt: Union[sqlparse.sql.IdentifierList, sqlparse.sql.Identifier]
                           ) -> Tuple[Dict[str, str], List[str]]:
        assert isinstance(sub_stmt, sqlparse.sql.Identifier) or isinstance(sub_stmt, sqlparse.sql.IdentifierList)
        alias_to_real_name = {}
        real_names = []

        xs = filter(lambda x: type(x) == sqlparse.sql.Identifier, sub_stmt) \
            if isinstance(sub_stmt, sqlparse.sql.IdentifierList) \
            else [sub_stmt]

        for item in xs:
            assert isinstance(item, sqlparse.sql.Identifier)

            alias = item.get_alias() if item.has_alias() else None
            real_name = item.get_real_name()

            if alias is not None:
                alias_to_real_name[alias] = real_name

            real_names += [real_name]

        return alias_to_real_name, real_names

    def destruct(sub_stmt: sqlparse.sql.Where, alias_to_real_name: Dict[str, str]):
        joins = set()
        ranges: dict = {}
        predicates: dict = {}
        legal_operators = ['<', '>', '=']
        items = filter(lambda token: isinstance(token, sqlparse.sql.Comparison), sub_stmt)

        for item in items:
            assert isinstance(item, sqlparse.sql.Comparison)

            l = item.left
            assert isinstance(l, sqlparse.sql.Identifier)
            r = item.right

            assert len(l.tokens) == 3, "only support to '<table(Alias)>.<attr>'"
            # l.tokens is like: [Name, Punctuation, Name], e.g: Stu.grade -> ['Stu', '.', 'grade']

            # the first 'Name' may be an alias.
            table1_name = alias_to_real_name[l[0].value] if l[0].value in alias_to_real_name else l[0].value
            attr1_name = l[2].value
            if isinstance(r, sqlparse.sql.Identifier):
                symbol = r.value
                if symbol.startswith('"') and symbol.endswith('"'):
                    raise NotImplementedError("not support 'varchar' yet.")
                else:
                    assert len(r.tokens) == 3, "attr should have a fully name, e.g. replace 'age' with 'Std.age'."
                    table2_name = alias_to_real_name[r[0].value] if r[0].value in alias_to_real_name else r[0].value
                    attr2_name = r[2].value
                    joins.add(
                        ((attr1_name, table1_name), (attr2_name, table2_name))
                    )

            elif isinstance(r, sqlparse.sql.Token):
                value = float(r.value)

                op = item.value.removeprefix(l.value).removesuffix(r.value).strip()
                assert op in legal_operators, "op must be the one of ['<','=','>']. "

                range_: Tuple = (Bound.Lower, Bound.Upper)

                """
                the real-world complex sql may contain duplicate predicates for a single attribute, 
                like: 'WHERE age > 18 AND age < 24', it means a RANGE.
                we don't consider some scenarios, like:
                    - WHERE age > 18 AND age < 24 AND age > 27 AND age < 30 (multi range) 
                    - WHERE age > 24 AND age < 18 (No length range)
                """
                if (attr1_name, table1_name) in ranges.keys():
                    range_ = ranges[(attr1_name, table1_name)]

                lower, upper = 0, 1
                if op == "=":
                    range_ = (value, value)
                if op == "<":
                    range_ = (range_[lower], value)
                if op == ">":
                    range_ = (value, range_[upper])

                # (able_attr, table_name) : (LB, UB)
                ranges[(attr1_name, table1_name)] = range_

                if (attr1_name, table1_name) not in predicates:
                    predicates[(attr1_name, table1_name)] = [(op, value)]
                else:
                    predicates[(attr1_name, table1_name)] += [(op, value)]

        return joins, ranges, predicates

    parsed_sql = sp.parse(query)
    stmt = parsed_sql[0]  # only single query.

    """
        why assert 'len(stmt.tokens) == 9':
        
             [1]      [3]  [5]         [7]
              v        v    v           v
        SELECT|COUNT(*)|FROM|Std, Course|WHERE Std.id = Course.id AND age < 18
        ^^^^^^ ^^^^^^^^ ^^^^ ^^^^^^^^^^^ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
          [0]    [2]     [4]     [6]           [8]
    """
    assert len(stmt.tokens) == 9, ""

    table_tokens = stmt[-3]  # IdentifierList, iterable.
    dicts, real_table_names = get_name_and_alias(table_tokens)
    where_tokens = stmt[-1]  # Where, iterable.
    joins, ranges, predicates = destruct(where_tokens, dicts)

    return Query(tables=real_table_names, joins=joins, ranges=ranges, predicates=predicates)
