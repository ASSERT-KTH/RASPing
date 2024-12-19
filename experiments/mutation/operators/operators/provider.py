from .replace_rasp_comparison import ReplaceRaspComparisonOperator
from .modify_integer import IncrementInteger
from .modify_integer import DecrementInteger
from .modify_rasp_indices import IncrementRaspIndices
from .modify_rasp_indices import DecrementRaspIndices
from .negate_rasp_sop import (
    NegateRaspSOpReturnStmt,
    NegateRaspSOpSelect,
    NegateRaspSOpAggregateValue,
    NegateRaspSOpConstructor,
)


class Provider:
    _operators = {
        "replace-rasp-comparison": ReplaceRaspComparisonOperator,
        "increment-integer": IncrementInteger,
        "decrement-integer": DecrementInteger,
        "increment-rasp-indices": IncrementRaspIndices,
        "decrement-rasp-indices": DecrementRaspIndices,
        "negate-rasp-sop-return-stmt": NegateRaspSOpReturnStmt,
        "negate-rasp-sop-select": NegateRaspSOpSelect,
        "negate-rasp-sop-aggregate-value": NegateRaspSOpAggregateValue,
        "negate-rasp-sop-constructor": NegateRaspSOpConstructor,
    }

    def __iter__(self):
        return iter(Provider._operators)

    def __getitem__(self, name):
        return Provider._operators[name]
