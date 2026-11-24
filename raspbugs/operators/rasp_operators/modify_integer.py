from cosmic_ray.operators.operator import Operator
import parso


class ModifyInteger(Operator):
    """An operator that modifies numeric constants."""

    VALUE = NotImplemented

    def mutation_positions(self, node):
        if isinstance(node, parso.python.tree.Number):
            yield (node.start_pos, node.end_pos)

    def mutate(self, node, index):
        """Modify the numeric value on `node`."""

        assert isinstance(node, parso.python.tree.Number)

        val = eval(node.value) + self.VALUE
        return parso.python.tree.Number(" " + str(val), node.start_pos)

    @classmethod
    def examples(cls):
        return (
            ("1", "0"),
            ("1", "2"),
        )


class IncrementInteger(ModifyInteger):
    """An operator that increments numeric constants."""

    VALUE = 1


class DecrementInteger(ModifyInteger):
    """An operator that decrements numeric constants."""

    VALUE = -1
