from tracr.rasp import rasp

def make_count_occurrences() -> rasp.SOp:
    """
    Creates an SOp that transforms a sequence so each element is replaced by the number of times it appears in the sequence.
    """
    # Selector that compares each element with every other element to find duplicates
    eq_selector = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.EQ).named("eq_selector")

    # Counts the occurrences of each element based on the equality comparison
    count_occurrences = rasp.SelectorWidth(eq_selector).named("count_occurrences")

    return count_occurrences
