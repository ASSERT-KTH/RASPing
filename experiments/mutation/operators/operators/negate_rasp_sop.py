from cosmic_ray.operators.operator import Operator
import parso
import parso.python
import parso.python.tree


class NegateRaspSOpReturnStmt(Operator):
    """An operator that negates SOps."""

    def mutation_positions(self, node):
        if isinstance(node, parso.python.tree.ReturnStmt):
            yield (node.start_pos, node.end_pos)

    def mutate(self, node, index):
        """Modify the node by multypling it by -1."""
        assert isinstance(node, parso.python.tree.ReturnStmt)
        node.children.extend([parso.parse(" * -1")])
        return node

    @classmethod
    def examples(cls):
        return (
            (
                'return rasp.SelectorWidth(all_true_selector).named("length")',
                'return rasp.SelectorWidth(all_true_selector).named("length") * -1',
            ),
        )


class NegateRaspSOpSelect(Operator):
    """An operator that negates SOps provided as arguments to rasp.Select."""

    def mutation_positions(self, node):
        if node.type == "atom_expr" and node.get_code().strip().startswith(
            "rasp.Select("
        ):
            # We change the first two arguments (query, key).
            for _ in range(2):
                yield (node.start_pos, node.end_pos)

    def mutate(self, node, index):
        """Modify the node by multypling the query and key arguments by -1."""
        assert node.type == "atom_expr"
        assert node.get_code().strip().startswith("rasp.Select(")
        arglist = node.children[2].children[1]
        idx = 0 if index == 0 else 2
        # if we are dealing with a PythonNode, we need to add a new child
        if isinstance(arglist.children[idx], parso.python.tree.PythonNode):
            arglist.children[idx].children.extend([parso.parse(" * -1")])
        # if we are dealing with a Name, we need to change the value
        elif isinstance(arglist.children[idx], parso.python.tree.Name):
            arglist.children[idx] = parso.parse(
                arglist.children[idx].get_code() + " * -1"
            )
        else:
            raise NotImplementedError(f"Unexpected type: {type(arglist.children[idx])}")
        return node

    @classmethod
    def examples(cls):
        return (
            (
                'rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.TRUE).named("all_true_selector")',
                'rasp.Select(-1 * rasp.tokens, rasp.tokens, rasp.Comparison.TRUE).named("all_true_selector")',
            ),
            (
                'rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.TRUE).named("all_true_selector")',
                'rasp.Select(rasp.tokens, -1 * rasp.tokens, rasp.Comparison.TRUE).named("all_true_selector")',
            ),
        )


class NegateRaspSOpAggregateValue(Operator):
    """An operator that negates SOps provided as arguments to rasp.Aggregate."""

    def mutation_positions(self, node):
        if node.type == "atom_expr" and node.get_code().strip().startswith(
            "rasp.Aggregate("
        ):
            yield (node.start_pos, node.end_pos)

    def mutate(self, node, index):
        """Modify the node by multypling the value argument by -1."""
        assert node.type == "atom_expr"
        assert node.get_code().strip().startswith("rasp.Aggregate(")
        arglist = node.children[2].children[1]
        idx = 2  # We are changing the second argument (value)
        # if we are dealing with a PythonNode, we need to add a new child
        if isinstance(arglist.children[idx], parso.python.tree.PythonNode):
            arglist.children[idx].children.extend([parso.parse(" * -1")])
        # if we are dealing with a Name, we need to change the value
        elif isinstance(arglist.children[idx], parso.python.tree.Name):
            arglist.children[idx] = parso.parse(
                arglist.children[idx].get_code() + " * -1"
            )
        else:
            raise NotImplementedError(f"Unexpected type: {type(arglist.children[idx])}")
        return node

    @classmethod
    def examples(cls):
        return (
            (
                'rasp.Aggregate(reverse_selector, sop).named("reverse")',
                'rasp.Aggregate(reverse_selector, sop * -1).named("reverse")',
            ),
        )


class NegateRaspSOpConstructor(Operator):
    """An operator that negates the result of an SOp constructor."""

    SOPs = {
        "rasp.SelectorWidth(",
        "rasp.Aggregate(",
        "rasp.numerical(",
        "rasp.SequenceMap(",
        "rasp.Map(",
        "rasp.Full(",
    }

    def mutation_positions(self, node):
        if node.type == "atom_expr" and any(
            node.get_code().strip().startswith(x) for x in self.SOPs
        ):
            yield (node.start_pos, node.end_pos)

    def mutate(self, node, index):
        """Modify the node by multypling an SOp by -1."""
        assert node.type == "atom_expr"
        assert any(node.get_code().strip().startswith(x) for x in self.SOPs)
        node.children.extend([parso.parse(" * -1")])
        return node

    @classmethod
    def examples(cls):
        return (
            (
                'rasp.Aggregate(reverse_selector, sop).named("reverse")',
                'rasp.Aggregate(reverse_selector, sop).named("reverse") * -1',
            ),
        )
