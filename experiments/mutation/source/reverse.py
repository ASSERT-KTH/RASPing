from tracr.rasp import rasp


def make_length() -> rasp.SOp:
    """Creates the `length` SOp using selector width primitive.

    Example usage:
      length = make_length()
      length("abcdefg")
      >> [7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0]

    Returns:
      length: SOp mapping an input to a sequence, where every element
        is the length of that sequence.
    """
    all_true_selector = rasp.Select(
        rasp.tokens, rasp.tokens, rasp.Comparison.TRUE
    ).named("all_true_selector")
    return rasp.SelectorWidth(all_true_selector).named("length")


length = make_length()


def make_reverse(sop: rasp.SOp) -> rasp.SOp:
    """Create an SOp that reverses a sequence, using length primitive.

    Example usage:
      reverse = make_reverse(rasp.tokens)
      reverse("Hello")
      >> ['o', 'l', 'l', 'e', 'H']

    Args:
      sop: an SOp

    Returns:
      reverse : SOp that reverses the input sequence.
    """
    opp_idx = (length - rasp.indices).named("opp_idx")
    opp_idx = (opp_idx - 1).named("opp_idx-1")
    reverse_selector = rasp.Select(rasp.indices, opp_idx, rasp.Comparison.EQ).named(
        "reverse_selector"
    )
    return rasp.Aggregate(reverse_selector, sop).named("reverse")
