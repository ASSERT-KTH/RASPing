from cosmic_ray.operators.operator import Operator
import parso
import parso.python
import parso.python.tree


class ModifyRaspIndices(Operator):
    """An operator that modifies rasp.indices expressions."""

    EXPRESSION = NotImplemented

    def mutation_positions(self, node):
        if isinstance(node, parso.python.tree.PythonNode):
            if node.type == "atom_expr" and node.get_code().strip() == "rasp.indices":
                yield (node.start_pos, node.end_pos)

    def mutate(self, node, index):
        """Modify the node by adding the expression."""
        assert isinstance(node, parso.python.tree.PythonNode)
        node.children.extend([parso.parse(self.EXPRESSION)])
        return node

    @classmethod
    def examples(cls):
        return (
            ("rasp.indices", "rasp.indices + 1"),
            ("rasp.indices", "rasp.indices - 1"),
        )


class IncrementRaspIndices(ModifyRaspIndices):
    """An operator that increments rasp.indices expressions."""

    EXPRESSION = " + 1"


class DecrementRaspIndices(ModifyRaspIndices):
    """An operator that decrements rasp.indices expressions."""

    EXPRESSION = " - 1"
