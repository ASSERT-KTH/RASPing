from tracr.rasp import rasp


def make_sort_unique(vals: rasp.SOp, keys: rasp.SOp) -> rasp.SOp:
    """Returns vals sorted by < relation on keys.

    Only supports unique keys.

    Example usage:
      sort = make_sort(rasp.tokens, rasp.tokens)
      sort([2, 4, 3, 1])
      >> [1, 2, 3, 4]

    Args:
      vals: Values to sort.
      keys: Keys for sorting.
    """
    smaller = rasp.Select(keys, keys, rasp.Comparison.LT).named("smaller")
    target_pos = rasp.SelectorWidth(smaller).named("target_pos")
    sel_new = rasp.Select(target_pos, rasp.indices, rasp.Comparison.EQ)
    return rasp.Aggregate(sel_new, vals).named("sort")


def make_sort(
    vals: rasp.SOp, keys: rasp.SOp, *, max_seq_len: int, min_key: float
) -> rasp.SOp:
    """Returns vals sorted by < relation on keys, which don't need to be unique.

    The implementation differs from the RASP paper, as it avoids using
    compositions of selectors to break ties. Instead, it uses the arguments
    max_seq_len and min_key to ensure the keys are unique.

    Note that this approach only works for numerical keys.

    Example usage:
      sort = make_sort(rasp.tokens, rasp.tokens, 5, 1)
      sort([2, 4, 3, 1])
      >> [1, 2, 3, 4]
      sort([2, 4, 1, 2])
      >> [1, 2, 2, 4]

    Args:
      vals: Values to sort.
      keys: Keys for sorting.
      max_seq_len: Maximum sequence length (used to ensure keys are unique)
      min_key: Minimum key value (used to ensure keys are unique)

    Returns:
      Output SOp of sort program.
    """
    keys = rasp.SequenceMap(
        lambda x, i: x + min_key * i / max_seq_len, keys, rasp.indices
    )
    return make_sort_unique(vals, keys)
