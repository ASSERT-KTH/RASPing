from tracr.rasp import rasp

def make_count_less_than() -> rasp.SOp:
    # Create a selector that identifies where one element is less than another in the sequence.
    lt_selector = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.LT).named("lt_selector")

    # Count the number of True comparisons (i.e., number of elements less than) for each element.
    count_lt = rasp.SelectorWidth(lt_selector).named("count_lt")

    return count_lt
