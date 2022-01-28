"""Classes and functions to read Alpino parse trees and determine what the heads are in EventDNA annotations."""

import xml.etree.ElementTree as ET
from pathlib import Path
from pprint import pprint


class AlpinoTreeHandler:
    """This class reads in an alpino .xml file and provides methods to find which tokens are heads."""

    def __init__(self, alpino_file):
        self.tree = ET.parse(alpino_file)
        self.node_to_parent_map = {c: p for p in self.tree.iter() for c in p}

    def get_parents(self, node):
        """Return a list such that the first element is the queried node and the next are all its ancestor nodes, up to the root of the tree."""
        parents = [node]
        while True:
            last_node = parents[-1]
            if last_node in self.node_to_parent_map:
                parents.append(self.node_to_parent_map[last_node])
            else:
                break
        return parents

    def find_head_leaves(self):
        """Depth-search through the tree. Only leaves are returned."""

        ## Negative restriction
        # Travel through all nodes in the tree in depth-first fashion, starting at the root.
        # Eliminate all those nodes that don't conform to the requirements.

        # We will add nodes to `to_visit` as we travel in the tree.
        to_visit = [self.tree.getroot()]
        found = []
        while True:

            ## debug
            # node_repr = (
            #     lambda node: node.get("rel")
            #     if not node.get("word")
            #     else node.get("word")
            # )
            # print([node_repr(n) for n in to_visit], [node_repr(n) for n in found])

            # If all nodes in the tree have been travelled, stop the loop.
            if len(to_visit) == 0:
                break

            # Pick the next node in the to_visit list and remove it from that list.
            current_node = to_visit.pop(0)

            # Determine whether the search for heads must stop here or continue.
            def check_stop(node):
                """If a node gets a True check here, search stops at that node and doesn't travel deeper in the tree."""
                # ! to allow all heads, comment out the following two conditions
                # if node.get("cat") in ["ap", "advp", "pp"]:  # "pp", "advp"
                #     return True
                # if node.get("rel") == "mod":
                #     return True
                return False

            if check_stop(current_node):
                continue  # NB: continue means skip to the next iteration of the "while true" loop.

            # If the current node passes the check_stop test and is a leaf, add it to the `found` list.
            # Else, take the children of the current node and add them to the to_visit list.
            is_leaf = lambda node: len(list(node)) == 0
            if is_leaf(current_node):
                found.append(current_node)
            else:
                kids = [n for n in current_node]
                to_visit.extend(kids)
                continue

        ## Positive restriction
        # Go over the `found` list and filter out all nodes that DON'T have a head somewhere in their ancestry.

        def has_hd_ancestor(node):
            """True if any of the query node's ancestors is a head node."""
            parents = self.get_parents(node)
            for p in parents:
                if p.get("rel") == "hd":
                    return True
            return False

        found = [n for n in found if has_hd_ancestor(n)]

        return found

    def head_vector(self):
        """Given an alpino tree, give a binary vector mapping over the tokens of the sentence described by the tree,
        such that 1 indicates that a token is part of a head node.
        e.g. [0, 1, 0, 1, 0, 0] --> tokens at index 1 and 3 are part of head nodes over the sentence.
        """

        def is_leaf(node):
            return len(list(node)) == 0

        # Get leaf nodes that are heads, as nodes.
        leaf_hd_nodes = self.find_head_leaves()

        # Get nodes that are tokens. These are always leaves.
        sentence_token_nodes = [
            node
            for node in self.tree.iter("node")
            if is_leaf(node) and node.get("word") is not None
        ]
        sentence_token_nodes = sorted(
            sentence_token_nodes,
            key=lambda node: int(
                node.get("begin")
            ),  # don't forget to int() --> else the numbers are strings
        )
        # Sanity check: the sentence found by ordering the nodes is equal to the sentence given as a Sentence element in the xml.
        # For unknown reasons a None node is added to the list. This naively removes it (CC 13/05/2019).
        tokens_from_sentence_nodes = [
            node.get("word") for node in sentence_token_nodes
        ]
        x_tokens_by_sentence = self.tree.findall("./sentence")[0].text.split()
        assert (
            tokens_from_sentence_nodes == x_tokens_by_sentence
        ), "{} != {}".format(tokens_from_sentence_nodes, x_tokens_by_sentence)

        # Collect information: go over token nodes and show 1 if the token is part of the list of head tokens and 0 otherwise.
        head_flags = [
            (1 if node in leaf_hd_nodes else 0)
            for node in sentence_token_nodes
        ]
        assert len(tokens_from_sentence_nodes) == len(
            head_flags
        )  # Sanity check.

        return head_flags


def add_heads(dnaf, alpino_dir) -> None:
    """Add head set info to the event annotations found in the given DNAF."""

    # Get a dict of sentence numbers to the correct head vector.
    head_vector_map = {
        int(file.stem): AlpinoTreeHandler(file).head_vector()
        for file in alpino_dir.iterdir()
    }

    # Go over all events in the dnaf document.
    for _, event in dnaf["doc"]["annotations"]["events"].items():

        home_sentence_id = event["home_sentence"]

        # Build vector for this event annotation over the sentence.
        # EG. "[President Trump addressed Congress] ." --> [1, 1, 1, 1, 0]
        sentence_tokens = sorted(
            dnaf["doc"]["sentences"][home_sentence_id]["token_ids"]
        )  # Tokens are represented as indices.
        event_tokens = sorted(event["features"]["span"])
        event_over_sentence_vector = [
            (1 if st in event_tokens else 0) for st in sentence_tokens
        ]

        # Fetch vector of all heads over the sentence from the head_vector_map defined previously.
        # eg. "President Trump addressed Congress ." --> [1, 0, 0, 1, 0]
        sentence_number = int(
            home_sentence_id.split("_")[1]
        )  # from e.g. "sentence_2" to 2
        head_over_sentence_vector = head_vector_map[sentence_number]

        # Get the overlap between head vector and sentence vector to get a set of tokens that are heads in an annotation.
        head_set = {
            i
            for i, (val1, val2) in enumerate(
                zip(event_over_sentence_vector, head_over_sentence_vector)
            )
            if val1 == val2 == 1
        }  # the `== 1` is there so as not to count the 0 values also.

        # Write the resulting head set as additional info to the DNAF.
        event["head_set"] = head_set
