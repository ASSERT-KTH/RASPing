from tracr.rasp import rasp

def make_histogram() -> rasp.SOp:
    """
    Creates an SOp that calculates the histogram of elements in the input sequence.
    For each element in the sequence, it returns the count of that element in the sequence.
    """
    # Create a selector matrix indicating where elements in the sequence are equal.
    equal_selector = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.EQ).named("equal_selector")

    # Count the number of times each element appears in the sequence.
    histogram_count = rasp.SelectorWidth(equal_selector).named("histogram_count")

    return histogram_count
