from tracr.rasp import rasp
from typing import List


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


def make_frac_prevs(bools: rasp.SOp) -> rasp.SOp:
    """Count the fraction of previous tokens where a specific condition was True.

     (As implemented in the RASP paper.)

    Example usage:
      num_l = make_frac_prevs(rasp.tokens=="l")
      num_l("hello")
      >> [0, 0, 1/3, 1/2, 2/5]

    Args:
      bools: SOp mapping a sequence to a sequence of booleans.

    Returns:
      frac_prevs: SOp mapping an input to a sequence, where every element
        is the fraction of previous "True" tokens.
    """
    bools = rasp.numerical(bools)
    prevs = rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.LEQ)
    return rasp.numerical(rasp.Aggregate(prevs, bools, default=0)).named("frac_prevs")


def make_pair_balance(sop: rasp.SOp, open_token: str, close_token: str) -> rasp.SOp:
    """Return fraction of previous open tokens minus the fraction of close tokens.

     (As implemented in the RASP paper.)

    If the outputs are always non-negative and end in 0, that implies the input
    has balanced parentheses.

    Example usage:
      num_l = make_pair_balance(rasp.tokens, "(", ")")
      num_l("a()b(c))")
      >> [0, 1/2, 0, 0, 1/5, 1/6, 0, -1/8]

    Args:
      sop: Input SOp.
      open_token: Token that counts positive.
      close_token: Token that counts negative.

    Returns:
      pair_balance: SOp mapping an input to a sequence, where every element
        is the fraction of previous open tokens minus previous close tokens.
    """
    bools_open = rasp.numerical(sop == open_token).named("bools_open")
    opens = rasp.numerical(make_frac_prevs(bools_open)).named("opens")

    bools_close = rasp.numerical(sop == close_token).named("bools_close")
    closes = rasp.numerical(make_frac_prevs(bools_close)).named("closes")

    pair_balance = rasp.numerical(rasp.LinearSequenceMap(opens, closes, 1, -1))
    return pair_balance.named("pair_balance")


def make_shuffle_dyck(pairs: List[str]) -> rasp.SOp:
    """Returns 1 if a set of parentheses are balanced, 0 else.

     (As implemented in the RASP paper.)

    Example usage:
      shuffle_dyck2 = make_shuffle_dyck(pairs=["()", "{}"])
      shuffle_dyck2("({)}")
      >> [1, 1, 1, 1]
      shuffle_dyck2("(){)}")
      >> [0, 0, 0, 0, 0]

    Args:
      pairs: List of pairs of open and close tokens that each should be balanced.
    """
    assert len(pairs) >= 1

    # Compute running balance of each type of parenthesis
    balances = []
    for pair in pairs:
        assert len(pair) == 2
        open_token, close_token = pair
        balance = make_pair_balance(
            rasp.tokens, open_token=open_token, close_token=close_token
        ).named(f"balance_{pair}")
        balances.append(balance)

    # Check if balances where negative anywhere -> parentheses not balanced
    any_negative = balances[0] < 0
    for balance in balances[1:]:
        any_negative = any_negative | (balance < 0)

    # Convert to numerical SOp
    any_negative = rasp.numerical(rasp.Map(lambda x: x, any_negative)).named(
        "any_negative"
    )

    select_all = rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.TRUE).named(
        "select_all"
    )
    has_neg = rasp.numerical(rasp.Aggregate(select_all, any_negative, default=0)).named(
        "has_neg"
    )

    # Check if all balances are 0 at the end -> closed all parentheses
    all_zero = balances[0] == 0
    for balance in balances[1:]:
        all_zero = all_zero & (balance == 0)

    select_last = rasp.Select(rasp.indices, length - 1, rasp.Comparison.EQ).named(
        "select_last"
    )
    last_zero = rasp.Aggregate(select_last, all_zero).named("last_zero")

    not_has_neg = (~has_neg).named("not_has_neg")
    return (last_zero & not_has_neg).named("shuffle_dyck")
