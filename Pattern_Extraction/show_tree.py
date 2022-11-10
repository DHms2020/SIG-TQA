import sys
import copy
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Node:
    def __init__(self, sym ="", type=""):
        self.sym = sym
        self.type = type


class Tree:
    def __init__(self, tree):
        self.root = self.gen_tree(tree)
        self.view = self.tree_view(self.root, 0)


    def gen_tree(self, seq):
        if seq.count("(") != seq.count(")"):
            return None
        if seq.count("(") == 0:
            if len(seq.strip()) == 0:
                return None
            return Node(sym=seq.strip())
        if seq[0] != "(" or seq[-1] != ")":
            return None
        tree = []
        sub = ""
        for c in seq[1: -1].strip() + "^":
            if c in [" ", "\t", "(", ")", "^"]:
                if sub.count("(") == sub.count(")"):
                    node = self.gen_tree(sub)
                    if node is not None:
                        tree.append(node)
                    sub = ""
            sub += c
        return tree


    def tree_view(self, tree, level=0):
        if isinstance(tree, Node):
            return "    " * level + tree.sym
        lines = "    " * level + tree[0].sym
        for x in tree[1:]:
            lines += "\n" + self.tree_view(x, level + 1)
        return lines