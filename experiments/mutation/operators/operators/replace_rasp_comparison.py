from cosmic_ray.operators.operator import Operator
import parso
import parso.python
import parso.python.tree


class ReplaceRaspComparisonOperator(Operator):
    """An operator that modifies RASP Comparison nodes."""

    # FIXME: use ordered set to improve performance if needed
    OPERATORS = [
        "rasp.Comparison.EQ",
        "rasp.Comparison.LT",
        "rasp.Comparison.LEQ",
        "rasp.Comparison.GT",
        "rasp.Comparison.GEQ",
        "rasp.Comparison.NEQ",
        "rasp.Comparison.TRUE",
        "rasp.Comparison.FALSE",
    ]

    def mutation_positions(self, node):
        if isinstance(node, parso.python.tree.PythonNode):
            if (
                node.type == "atom_expr"
                and len(node.children) == 3
                and isinstance(node.children[0], parso.python.tree.Name)
                and node.get_code().strip() in self.OPERATORS
            ):
                # There are seven different alternatives to each operator,
                # so we must return a mutation position for each of them
                for _ in range(7):
                    yield (node.start_pos, node.end_pos)

    def mutate(self, node, index):
        """Modify the comparison operator."""
        idx = self.OPERATORS.index(node.get_code().strip())
        if index == idx:
            index += 1
        node.children[2].children[1].value = self.OPERATORS[index].split(".")[-1]
        return node

    @classmethod
    def examples(cls):
        return (("rasp.Comparison.TRUE", "rasp.Comparison.FALSE"),)
