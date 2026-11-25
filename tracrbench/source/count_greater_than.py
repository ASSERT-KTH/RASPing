from tracr.rasp import rasp

def make_count_greater_than() -> rasp.SOp:
    # Creating a selector that identifies elements greater than each element.
    greater_than_selector = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.GT).named("greater_than_selector")

    # Counting the number of elements greater than each element.
    count_greater_than = rasp.SelectorWidth(greater_than_selector).named("count_greater_than")

    return count_greater_than
