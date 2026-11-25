from tracr.rasp import rasp

def make_count_frequency() -> rasp.SOp:
    # Create a comparison matrix where each element is compared to every other element for equality.
    equality_selector = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.EQ).named("equality_selector")

    # Use SelectorWidth to count the frequency of each element based on the equality comparison matrix.
    frequency_count = rasp.SelectorWidth(equality_selector).named("frequency_count")

    # The result is a sequence where each element is replaced by its frequency count.
    return frequency_count
