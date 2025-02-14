from tracr.rasp import rasp


def make_hist() -> rasp.SOp:
    """Returns the number of times each token occurs in the input.

     (As implemented in the RASP paper.)

    Example usage:
      hist = make_hist()
      hist("abac")
      >> [2, 1, 2, 1]
    """
    same_tok = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.EQ).named(
        "same_tok"
    )
    return rasp.SelectorWidth(same_tok).named("hist")


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
    vals: rasp.SOp, keys: rasp.SOp, /, max_seq_len: int, min_key: float
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


def make_sort_freq(max_seq_len: int) -> rasp.SOp:
    """Returns tokens sorted by the frequency they appear in the input.

    Tokens the appear the same amount of times are output in the same order as in
    the input.

    Example usage:
      sort = make_sort_freq(rasp.tokens, rasp.tokens, 5)
      sort([2, 4, 2, 1])
      >> [2, 2, 4, 1]

    Args:
      max_seq_len: Maximum sequence length (used to ensure keys are unique)
    """
    hist = -1 * make_hist().named("hist")
    return make_sort(rasp.tokens, hist, max_seq_len=max_seq_len, min_key=1).named(
        "sort_freq"
    )
