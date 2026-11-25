from tracr.rasp import rasp

def make_rank() -> rasp.SOp:
    # Selector that creates a comparison matrix where each element is compared to every other element to find how many are smaller.
    less_than_selector = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.LT).named("less_than_selector")

    # Count the number of elements that are smaller than each element.
    smaller_count = rasp.SelectorWidth(less_than_selector).named("smaller_count")

    # Since ranks start from 1, add 1 to each count to get the rank.
    rank = rasp.Map(lambda x: x + 1, smaller_count).named("rank")

    return rank
