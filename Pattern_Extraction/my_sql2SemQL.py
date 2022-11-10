import argparse
import json
import sys

import copy
from utils import load_dataSets
from show_tree import Tree
from collections import defaultdict
import pickle

from src.rule.my_semQL import Root1, Root, A, C, T, Sel, Sup, Filter, Order


Keywords = ['des', 'asc', 'and', 'or', 'sum', 'min', 'max', 'avg', 'none', '=', '!=', '<', '>', '<=', '>=', 'between',
            'like', 'not_like'] + [
               'in', 'not_in', 'count', 'intersect', 'union', 'except'
           ]

CMP_OP=['=', '!=', '<', '>', '<=', '>=']
AGG_OP=['sum', 'min', 'max', 'avg']
CONDI_1_OP=['in', 'not_in']
CONDI_2_OP=['like', 'not_like']
ORDER_OP=['des', 'asc']

NONE_Terminal_DICT={'CMP_OP':CMP_OP,'AGG_OP':AGG_OP,'CONDI_1_OP':CONDI_1_OP,'CONDI_2_OP':CONDI_2_OP,'ORDER_OP':ORDER_OP}

SWITCH_DICT = {type(1.0): 'number', type('string'): 'text', type([1,2,3]): 'othercol'}
count_dict=defaultdict(int)

def element2noneterminal(symbol):
    for key,value in NONE_Terminal_DICT.items():
        if symbol in value :
            if key in ['CMP_OP','AGG_OP']:
                return key,symbol
            return key,None
    return symbol,None

def add_child_father(Child, Father):
    for x in Child.production.split(' ')[1:]:
        if x in Keywords:
            Child.children.append(x)
    Child.set_parent(Father)
    Father.add_children(Child)


class Parser:
    def __init__(self):
        self.copy_selec = None
        self.sel_result = []
        self.colSet = set()

    def _init_rule(self):
        self.copy_selec = None
        self.colSet = set()

    def _parse_root(self, sql, F_Node):
        """
        parsing the sql by the grammar
        R ::= Select | Select Filter | Select Order | ... |
        :return: [R(), states]
        """
        use_sup, use_ord, use_fil = True, True, False

        if sql['sql']['limit'] == None:
            use_sup = False

        if sql['sql']['orderBy'] == []:
            use_ord = False
        elif sql['sql']['limit'] != None:
            use_ord = False

        # check the where and having
        if sql['sql']['where'] != [] or \
                sql['sql']['having'] != []:
            use_fil = True

        if use_fil and use_sup:
            Node = Root(0)
            add_child_father(Node, F_Node)
            return [Node], ['FILTER', 'SUP', 'SEL']
        elif use_fil and use_ord:
            Node = Root(1)
            add_child_father(Node, F_Node)
            return [Node], ['ORDER', 'FILTER', 'SEL']
        elif use_sup:
            Node = Root(2)
            add_child_father(Node, F_Node)
            return [Node], ['SUP', 'SEL']
        elif use_fil:
            Node = Root(3)
            add_child_father(Node, F_Node)
            return [Node], ['FILTER', 'SEL']
        elif use_ord:
            Node = Root(4)
            add_child_father(Node, F_Node)
            return [Node], ['ORDER', 'SEL']
        else:
            Node = Root(5)
            add_child_father(Node, F_Node)
            return [Node], ['SEL']

    def _parser_column0(self, sql, select, F_Node):
        """
        Find table of column '*'
        :return: T(table_id)
        """
        if len(sql['sql']['from']['table_units']) == 1:
            # count(*) from(select from ...)
            result=[]
            if type(sql['sql']['from']['table_units'][0][1])==dict:
                T_Node = T(-1)
                add_child_father(T_Node, F_Node)

                nest_query = {}
                nest_query['names'] = sql['names']
                nest_query['query_toks_no_value'] = ""
                nest_query['sql'] = sql['sql']['from']['table_units'][0][1]
                nest_query['col_table'] = sql['col_table']
                nest_query['col_set'] = sql['col_set']
                nest_query['table_names'] = sql['table_names']
                nest_query['question'] = sql['question']
                nest_query['query'] = sql['query']
                nest_query['keys'] = sql['keys']
                nest_query['col_types']=sql['col_types']
                re,Sub_ROOT=self.full_parse(nest_query)
                #为避免重复添加intersect/union/execept等，不调用函数
                # add_child_father(Sub_ROOT,T_Node)
                Sub_ROOT.set_parent(T_Node)
                T_Node.add_children(Sub_ROOT)

                result.extend(re)
                return result

            else:
                T_Node = T(sql['sql']['from']['table_units'][0][1],
                           tab_name=sql['table_names'][sql['sql']['from']['table_units'][0][1]])
                add_child_father(T_Node, F_Node)
                return T_Node
        else:
            table_list = []
            for tmp_t in sql['sql']['from']['table_units']:
                if type(tmp_t[1]) == int:
                    table_list.append(tmp_t[1])
            table_set, other_set = set(table_list), set()
            for sel_p in select:
                if sel_p[1][1][1] != 0:
                    other_set.add(sql['col_table'][sel_p[1][1][1]])

            if len(sql['sql']['where']) == 1:
                other_set.add(sql['col_table'][sql['sql']['where'][0][2][1][1]])
            elif len(sql['sql']['where']) == 3:
                other_set.add(sql['col_table'][sql['sql']['where'][0][2][1][1]])
                other_set.add(sql['col_table'][sql['sql']['where'][2][2][1][1]])
            elif len(sql['sql']['where']) == 5:
                other_set.add(sql['col_table'][sql['sql']['where'][0][2][1][1]])
                other_set.add(sql['col_table'][sql['sql']['where'][2][2][1][1]])
                other_set.add(sql['col_table'][sql['sql']['where'][4][2][1][1]])
            table_set = table_set - other_set
            if len(table_set) == 1:
                T_Node = T(list(table_set)[0], tab_name=sql['table_names'][list(table_set)[0]])
                add_child_father(T_Node, F_Node)
                return T_Node
            elif len(table_set) == 0 and sql['sql']['groupBy'] != []:
                T_Node = T(sql['col_table'][sql['sql']['groupBy'][0][1]],
                           tab_name=sql['table_names'][sql['col_table'][sql['sql']['groupBy'][0][1]]])
                add_child_father(T_Node, F_Node)
                return T_Node
            else:
                question = sql['question']
                self.sel_result.append(question)
                print('column * table error')
                T_Node = T(sql['sql']['from']['table_units'][0][1],
                           tab_name=sql['table_names'][sql['sql']['from']['table_units'][0][1]])
                add_child_father(T_Node, F_Node)
                return T_Node

    def _parse_select(self, sql, F_Node):
        """
        parsing the sql by the grammar
        Select ::= A | AA | AAA | ... |
        A ::= agg column table
        :return: [Sel(), states]
        """
        result = []
        select = sql['sql']['select'][1]

        number_a = len(select) - 1
        SEL_Node = Sel(number_a)
        add_child_father(Child=SEL_Node, Father=F_Node)
        result.append(SEL_Node)

        for sel in select:
            A_Node = A((sel[0]))
            add_child_father(A_Node, SEL_Node)
            result.append(A_Node)
            self.colSet.add(sql['col_set'].index(sql['names'][sel[1][1][1]]))

            C_Node = C(sql['col_set'].index(sql['names'][sel[1][1][1]]), col_name=sql['names'][sel[1][1][1]],col_type=sql['col_types'][sel[1][1][1]])
            add_child_father(C_Node, A_Node)
            result.append(C_Node)
            # now check for the situation with *
            if sel[1][1][1] == 0:
                result.append(self._parser_column0(sql, select, F_Node=A_Node))
            else:
                T_Node = T(sql['col_table'][sel[1][1][1]], tab_name=sql['table_names'][sql['col_table'][sel[1][1][1]]])
                add_child_father(T_Node, A_Node)
                result.append(T_Node)
            if not self.copy_selec:
                self.copy_selec = [copy.deepcopy(result[-2]), copy.deepcopy(result[-1])]

        return result, None

    def _parse_sup(self, sql, F_Node):
        """
        parsing the sql by the grammar
        Sup ::= Most A | Least A
        A ::= agg column table
        :return: [Sup(), states]
        """
        result = []
        select = sql['sql']['select'][1]
        if sql['sql']['limit'] == None:
            return result, None
        if sql['sql']['orderBy'][0] == 'desc':
            Su_Node = Sup(0)
            add_child_father(Su_Node, F_Node)
            result.append(Su_Node)
        else:
            Su_Node = Sup(1)
            add_child_father(Su_Node, F_Node)
            result.append(Su_Node)

        A_Node = A(sql['sql']['orderBy'][1][0][1][0])
        add_child_father(A_Node, Su_Node)
        result.append(A_Node)
        self.colSet.add(sql['col_set'].index(sql['names'][sql['sql']['orderBy'][1][0][1][1]]))
        C_Node = C(sql['col_set'].index(sql['names'][sql['sql']['orderBy'][1][0][1][1]]),
                   col_name=sql['names'][sql['sql']['orderBy'][1][0][1][1]],col_type=sql['col_types'][sql['sql']['orderBy'][1][0][1][1]])
        add_child_father(C_Node, A_Node)
        result.append(C_Node)
        if sql['sql']['orderBy'][1][0][1][1] == 0:
            result.append(self._parser_column0(sql, select, F_Node=A_Node))
        else:
            T_Node = T(sql['col_table'][sql['sql']['orderBy'][1][0][1][1]],
                       tab_name=sql['table_names'][sql['col_table'][sql['sql']['orderBy'][1][0][1][1]]])
            add_child_father(T_Node, A_Node)
            result.append(T_Node)
        return result, None

    def _parse_filter(self, sql, F_Node):
        """
        parsing the sql by the grammar
        Filter ::= and Filter Filter | ... |
        A ::= agg column table
        :return: [Filter(), states]
        """
        result = []
        # check the where
        if sql['sql']['where'] != [] and sql['sql']['having'] != []:
            Fil_first_Node = Filter(0)
            add_child_father(Fil_first_Node, F_Node)
            result.append(Fil_first_Node)

        if sql['sql']['where'] != []:
            # check the not and/or
            if len(sql['sql']['where']) == 1:
                result.extend(self.parse_one_condition(sql['sql']['where'][0], sql['names'], sql,
                                                       F_Node=Fil_first_Node if sql['sql']['having'] != [] else F_Node))
            elif len(sql['sql']['where']) == 3:
                if sql['sql']['where'][1] == 'or':
                    Fil_second_Node = Filter(1)
                    Father = Fil_first_Node if sql['sql']['having'] != [] else F_Node
                    add_child_father(Fil_second_Node, Father)
                    result.append(Fil_second_Node)
                else:
                    Fil_second_Node = Filter(0)
                    Father = Fil_first_Node if sql['sql']['having'] != [] else F_Node
                    add_child_father(Fil_second_Node, Father)
                    result.append(Fil_second_Node)
                    # result.append(Fil_second_Node)

                result.extend(
                    self.parse_one_condition(sql['sql']['where'][0], sql['names'], sql, F_Node=Fil_second_Node))
                result.extend(
                    self.parse_one_condition(sql['sql']['where'][2], sql['names'], sql, F_Node=Fil_second_Node))
            else:
                if sql['sql']['where'][1] == 'and' and sql['sql']['where'][3] == 'and':
                    Fil_second_Node = Filter(0)
                    Father = Fil_first_Node if sql['sql']['having'] != [] else F_Node
                    add_child_father(Fil_second_Node, Father)
                    result.append(Fil_second_Node)
                    result.extend(
                        self.parse_one_condition(sql['sql']['where'][0], sql['names'], sql, F_Node=Fil_second_Node))
                    Fil_third_Node = Filter(0)
                    add_child_father(Fil_third_Node, Fil_second_Node)
                    result.append(Fil_third_Node)
                    result.extend(
                        self.parse_one_condition(sql['sql']['where'][2], sql['names'], sql, F_Node=Fil_third_Node))
                    result.extend(
                        self.parse_one_condition(sql['sql']['where'][4], sql['names'], sql, F_Node=Fil_third_Node))
                elif sql['sql']['where'][1] == 'and' and sql['sql']['where'][3] == 'or':
                    Fil_second_Node = Filter(1)
                    Father = Fil_first_Node if sql['sql']['having'] != [] else F_Node
                    add_child_father(Fil_second_Node, Father)
                    result.append(Fil_second_Node)

                    Fil_third_Node = Filter(0)
                    add_child_father(Fil_third_Node, Fil_second_Node)
                    result.append(Fil_third_Node)

                    result.extend(
                        self.parse_one_condition(sql['sql']['where'][0], sql['names'], sql, F_Node=Fil_third_Node))
                    result.extend(
                        self.parse_one_condition(sql['sql']['where'][2], sql['names'], sql, F_Node=Fil_third_Node))
                    result.extend(
                        self.parse_one_condition(sql['sql']['where'][4], sql['names'], sql, F_Node=Fil_second_Node))
                elif sql['sql']['where'][1] == 'or' and sql['sql']['where'][3] == 'and':
                    Fil_second_Node = Filter(1)
                    Father = Fil_first_Node if sql['sql']['having'] != [] else F_Node
                    add_child_father(Fil_second_Node, Father)
                    result.append(Fil_second_Node)

                    Fil_third_Node = Filter(0)
                    add_child_father(Fil_third_Node, Fil_second_Node)
                    result.append(Fil_third_Node)
                    result.extend(
                        self.parse_one_condition(sql['sql']['where'][2], sql['names'], sql, F_Node=Fil_third_Node))
                    result.extend(
                        self.parse_one_condition(sql['sql']['where'][4], sql['names'], sql, F_Node=Fil_third_Node))
                    result.extend(
                        self.parse_one_condition(sql['sql']['where'][0], sql['names'], sql, F_Node=Fil_second_Node))
                else:
                    Fil_second_Node = Filter(1)
                    Father = Fil_first_Node if sql['sql']['having'] != [] else F_Node
                    add_child_father(Fil_second_Node, Father)
                    result.append(Fil_second_Node)

                    Fil_third_Node = Filter(1)
                    add_child_father(Fil_third_Node, Fil_second_Node)
                    result.append(Fil_third_Node)
                    result.extend(
                        self.parse_one_condition(sql['sql']['where'][0], sql['names'], sql, F_Node=Fil_second_Node))
                    result.extend(
                        self.parse_one_condition(sql['sql']['where'][2], sql['names'], sql, F_Node=Fil_third_Node))
                    result.extend(
                        self.parse_one_condition(sql['sql']['where'][4], sql['names'], sql, F_Node=Fil_third_Node))

        # check having
        if sql['sql']['having'] != []:
            Father = Fil_first_Node if sql['sql']['where'] != [] else F_Node
            result.extend(self.parse_one_condition(sql['sql']['having'][0], sql['names'], sql, F_Node=Father))
        return result, None

    def _parse_order(self, sql, F_Node):
        """
        parsing the sql by the grammar
        Order ::= asc A | desc A
        A ::= agg column table
        :return: [Order(), states]
        """
        result = []

        if 'order' not in sql['query_toks_no_value'] or 'by' not in sql['query_toks_no_value']:
            return result, None
        elif 'limit' in sql['query_toks_no_value']:
            return result, None
        else:
            if sql['sql']['orderBy'] == []:
                return result, None
            else:
                select = sql['sql']['select'][1]
                if sql['sql']['orderBy'][0] == 'desc':
                    Or_Node = Order(0)
                    add_child_father(Or_Node, F_Node)
                    result.append(Or_Node)
                else:
                    Or_Node = Order(1)
                    add_child_father(Or_Node, F_Node)
                    result.append(Or_Node)
                A_Node = A(sql['sql']['orderBy'][1][0][1][0])
                add_child_father(A_Node, Or_Node)
                result.append(A_Node)
                self.colSet.add(sql['col_set'].index(sql['names'][sql['sql']['orderBy'][1][0][1][1]]))
                C_Node = C(sql['col_set'].index(sql['names'][sql['sql']['orderBy'][1][0][1][1]]),
                           col_name=sql['names'][sql['sql']['orderBy'][1][0][1][1]],col_type=sql['col_types'][sql['sql']['orderBy'][1][0][1][1]])
                add_child_father(C_Node, A_Node)
                result.append(C_Node)
                if sql['sql']['orderBy'][1][0][1][1] == 0:
                    result.append(self._parser_column0(sql, select, F_Node=A_Node))
                else:
                    T_Node = T(sql['col_table'][sql['sql']['orderBy'][1][0][1][1]],
                               tab_name=sql['table_names'][sql['col_table'][sql['sql']['orderBy'][1][0][1][1]]])
                    add_child_father(T_Node, A_Node)
                    result.append(T_Node)
        return result, None

    def parse_one_condition(self, sql_condit, names, sql, F_Node):
        result = []
        # check if V(root)
        nest_query = True
        if type(sql_condit[3]) != dict:
            nest_query = False

        if sql_condit[0] == True:
        # for PCFG Method  pattern extraction
        # if sql_condit[0] == True and sql_condit[1] in [8,9]:
            if sql_condit[1] == 9:
                # not like only with values
                fil = Filter(10,value1=sql_condit[3])
                add_child_father(fil, F_Node)
            elif sql_condit[1] == 8:
                # not in with Root
                fil = Filter(19)
                add_child_father(fil, F_Node)
            else:
                print(sql_condit[1])
                raise NotImplementedError("not implement for the others FIL")
        else:
            # check for Filter (<,=,>,!=,between, >=,  <=, ...)
            single_map = {1: 8, 2: 2, 3: 5, 4: 4, 5: 7, 6: 6, 7: 3}
            nested_map = {1: 15, 2: 11, 3: 13, 4: 12, 5: 16, 6: 17, 7: 14}
            if sql_condit[1] in [1, 2, 3, 4, 5, 6, 7]:
                if nest_query == False:
                    if sql_condit[1] == 1:
                        fil = Filter(single_map[sql_condit[1]], value1=sql_condit[3],value2=sql_condit[4],schema_colname=sql['names'])
                        add_child_father(fil, F_Node)
                    else:
                        fil = Filter(single_map[sql_condit[1]],value1=sql_condit[3],schema_colname=sql['names'])
                        add_child_father(fil, F_Node)
                else:
                    if sql_condit[1] == 1:
                        fil = Filter(nested_map[sql_condit[1]],value2=sql_condit[4],schema_colname=sql['names'])
                        add_child_father(fil, F_Node)
                    else:
                        fil = Filter(nested_map[sql_condit[1]],schema_colname=sql['names'])
                        add_child_father(fil, F_Node)
            elif sql_condit[1] == 9:
                fil = Filter(9,value1=sql_condit[3],schema_colname=sql['names'])
                add_child_father(fil, F_Node)
            elif sql_condit[1] == 8:
                fil = Filter(18,schema_colname=sql['names'])
                add_child_father(fil, F_Node)
            else:
                print(sql_condit[1])
                raise NotImplementedError("not implement for the others FIL")

        result.append(fil)
        A_Node = A(sql_condit[2][1][0])
        add_child_father(A_Node, fil)
        result.append(A_Node)
        self.colSet.add(sql['col_set'].index(sql['names'][sql_condit[2][1][1]]))
        C_Node = C(sql['col_set'].index(sql['names'][sql_condit[2][1][1]]), col_name=sql['names'][sql_condit[2][1][1]],col_type=sql['col_types'][sql_condit[2][1][1]])
        add_child_father(C_Node, A_Node)
        result.append(C_Node)
        if sql_condit[2][1][1] == 0:
            select = sql['sql']['select'][1]
            result.append(self._parser_column0(sql, select, F_Node=A_Node))
        else:
            T_Node = T(sql['col_table'][sql_condit[2][1][1]],
                       tab_name=sql['table_names'][sql['col_table'][sql_condit[2][1][1]]])
            add_child_father(T_Node, A_Node)
            result.append(T_Node)

        # check for the nested value
        if type(sql_condit[3]) == dict:
            nest_query = {}
            nest_query['names'] = names
            nest_query['query_toks_no_value'] = ""
            nest_query['sql'] = sql_condit[3]
            nest_query['col_table'] = sql['col_table']
            nest_query['col_set'] = sql['col_set']
            nest_query['table_names'] = sql['table_names']
            nest_query['question'] = sql['question']
            nest_query['query'] = sql['query']
            nest_query['keys'] = sql['keys']
            nest_query['col_types']=sql['col_types']
            result.extend(self.parser(nest_query, F_Node=fil))

        return result

    def _parse_step(self, state, sql, F_Node):

        if state == 'ROOT':
            return self._parse_root(sql, F_Node=F_Node)

        if state == 'SEL':
            return self._parse_select(sql, F_Node=F_Node)

        elif state == 'SUP':
            return self._parse_sup(sql, F_Node=F_Node)

        elif state == 'FILTER':
            return self._parse_filter(sql, F_Node=F_Node)

        elif state == 'ORDER':
            return self._parse_order(sql, F_Node=F_Node)
        else:
            raise NotImplementedError("Not the right state")

    def full_parse(self, query):
        sql = query['sql']
        nest_query = {}
        nest_query['names'] = query['names']
        nest_query['query_toks_no_value'] = ""
        nest_query['col_table'] = query['col_table']
        nest_query['col_set'] = query['col_set']
        nest_query['table_names'] = query['table_names']
        nest_query['question'] = query['question']
        nest_query['query'] = query['query']
        nest_query['keys'] = query['keys']
        nest_query['col_types']=query['col_types']

        if sql['intersect']:
            Z = Root1(0)
            for x in Z.production.split(' ')[1:]:
                if x in Keywords:
                    Z.children.append(x)
            results = [Z]
            nest_query['sql'] = sql['intersect']
            results.extend(self.parser(query, F_Node=Z))
            results.extend(self.parser(nest_query,F_Node=Z))
            return results,Z

        if sql['union']:
            Z = Root1(1)
            for x in Z.production.split(' ')[1:]:
                if x in Keywords:
                    Z.children.append(x)
            results = [Z]
            nest_query['sql'] = sql['union']
            results.extend(self.parser(query, F_Node=Z))
            results.extend(self.parser(nest_query, F_Node=Z))
            return results,Z

        if sql['except']:
            Z = Root1(2)
            for x in Z.production.split(' ')[1:]:
                if x in Keywords:
                    Z.children.append(x)
            results = [Z]
            nest_query['sql'] = sql['except']
            results.extend(self.parser(query, F_Node=Z))
            results.extend(self.parser(nest_query, F_Node=Z))
            return results,Z

        Z = Root1(3)
        for x in Z.production.split(' ')[1:]:
            if x in Keywords:
                Z.children.append(x)
        results = [Z]
        results.extend(self.parser(query, F_Node=Z))

        return results, Z

    def parser(self, query, F_Node):
        stack = ["ROOT"]
        result = []
        while len(stack) > 0:
            state = stack.pop()
            if state == "ROOT":
                step_result, step_state = self._parse_step(state, query, F_Node=F_Node)
                R_Node = step_result[0]
            else:
                step_result, step_state = self._parse_step(state, query, F_Node=R_Node)
            result.extend(step_result)
            if step_state:
                stack.extend(step_state)
        return result

def new_LRD_search(ROOT):
    str=['(']
    cvalue_list=[]
    cmp_list=[]
    agg_list=[]
    str.append(ROOT)
    if isinstance(ROOT, C):
        str.append(ROOT.col_type)
        str.append(')')
        return str,[],[],[]
    elif isinstance(ROOT, T):
        if len(ROOT.children)==0:
            str.append('TABLE')
            str.append(')')
            return str,[],[],[]
    elif isinstance(ROOT, Filter):
        if ROOT.value1 != None:
            cvalue = ROOT.names[ROOT.value1[1]] if type(ROOT.value1) == list else ROOT.value1
            cvalue_list.append(cvalue)
            val=SWITCH_DICT.get(type(ROOT.value1))
            str.append(val)
        if ROOT.value2 != None:
            cvalue = ROOT.names[ROOT.value2[1]] if type(ROOT.value2) == list else ROOT.value2
            cvalue_list.append(cvalue)
            val=SWITCH_DICT.get(type(ROOT.value2))
            str.append(val)

    for element in ROOT.children:
        if element in Keywords:
            new_element,orielement = element2noneterminal(element)
            if orielement:
                if new_element == 'CMP_OP':
                    cmp_list.append(orielement)
                if new_element == 'AGG_OP':
                    agg_list.append(orielement)
            str.append(new_element)
        else:
            sub_str, sub_cvaluelist, sub_cmplist, sub_agglist = new_LRD_search(element)
            str.extend(sub_str)
            cvalue_list.extend(sub_cvaluelist)
            cmp_list.extend(sub_cmplist)
            agg_list.extend(sub_agglist)
    str.append(')')
    return str,cvalue_list,cmp_list,agg_list


def LRD_search(ROOT):
    str=['(']
    str.append(ROOT)
    if isinstance(ROOT, C):
        str.append(ROOT.col_type)
        str.append(')')
        return str
    elif isinstance(ROOT, T):
        if len(ROOT.children)==0:
            str.append('TABLE')
            str.append(')')
            return str
    elif isinstance(ROOT, Filter):
        if ROOT.value1 != None:
            val=SWITCH_DICT.get(type(ROOT.value1))
            str.append(val)
            count_dict[type(ROOT.value1)]+=1
        if ROOT.value2 != None:
            val=SWITCH_DICT.get(type(ROOT.value2))
            str.append(val)
            count_dict[type(ROOT.value2)] += 1

    for element in ROOT.children:
        if element in Keywords:
            new_element=element2noneterminal(element)
            str.append(new_element)
        else:
            str.extend(LRD_search(element))
    str.append(')')
    return str

def DLR_search(ROOT):
    str=[ROOT]
    if isinstance(ROOT,C):
        return str
    if isinstance(ROOT,T):
        return str
    for element in ROOT.children:
        if element in Keywords:
            continue
        else:
            str.extend(DLR_search(element))
    return str

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_path', type=str, help='dataset', required=True)
    arg_parser.add_argument('--table_path', type=str, help='table dataset', required=True)
    arg_parser.add_argument('--output', type=str, help='output data', required=True)
    args = arg_parser.parse_args()

    parser = Parser()

    # loading dataSets
    print(f"data_path is :{args.data_path}; table_path is :{args.table_path}")
    datas, table = load_dataSets(args)

    processed_data = []
    str_temp=[]

    max_len=0

    for i, d in enumerate(datas):
        temp_data={}
        if len(datas[i]['sql']['select'][1]) > 5:
            continue
        r, root = parser.full_parse(datas[i])
        # print("query : {}".format(datas[i]['question']))
        # print("SQL : {}".format((datas[i]['query'])))
        print(" ".join([str(x) for x in r]))
        LRD_str,cvalue_list,cmp_list,agg_list=new_LRD_search(root)
        print(" ".join([str(x) for x in LRD_str]))
        print("------tree {}-------".format(i))
        tree_str=" ".join([str(x) for x in LRD_str])
        Tree_info=Tree(tree_str)
        # print(Tree_info.view)

        temp_data['db_id']=d['db_id']
        temp_data['pattern']=tree_str
        temp_data['pattern_toks']=[str(x) for x in LRD_str]
        temp_data['query']=d['query']
        temp_data['query_toks']=d['query_toks']
        temp_data['query_toks_no_value']=d['query_toks_no_value']
        temp_data['question']=d['question']
        temp_data['question_toks']=d['question_toks']
        temp_data['sql']=d['sql']

        processed_data.append(temp_data)

    with open(args.output, 'w', encoding='utf8') as output:
        json.dump(processed_data,output)