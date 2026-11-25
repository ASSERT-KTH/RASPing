from tracr.rasp import rasp

def make_sort(vals: rasp.SOp, keys: rasp.SOp, *, max_seq_len: int, min_key: float) -> rasp.SOp:
    keys = rasp.SequenceMap(lambda x, i: x + min_key * i / max_seq_len, keys, rasp.indices)
    return make_sort_unique(vals, keys)

def make_sort_unique(vals: rasp.SOp, keys: rasp.SOp) -> rasp.SOp:
    smaller = rasp.Select(keys, keys, rasp.Comparison.LT).named("smaller")
    target_pos = rasp.SelectorWidth(smaller).named("target_pos")
    sel_new = rasp.Select(target_pos, rasp.indices, rasp.Comparison.EQ)
    return rasp.Aggregate(sel_new, vals).named("sort")

def make_length() -> rasp.SOp:
    all_true_selector = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.TRUE).named("all_true_selector")
    return rasp.SelectorWidth(all_true_selector).named("length")

def make_compute_median() -> rasp.SOp:
    # Sort the sequence.
    sorted_sequence = make_sort(rasp.tokens, rasp.tokens, max_seq_len=100, min_key=1)
    
    # Compute the length of the sequence.
    length = make_length()
    
    # Compute indices for the middle elements.
    middle1 = rasp.Map(lambda x: (x - 1) // 2, length)
    middle2 = rasp.Map(lambda x: x // 2, length)
    
    # Select middle elements based on computed indices.
    median1 = rasp.Aggregate(rasp.Select(rasp.indices, middle1, rasp.Comparison.EQ), sorted_sequence)
    median2 = rasp.Aggregate(rasp.Select(rasp.indices, middle2, rasp.Comparison.EQ), sorted_sequence)
    
    # Compute the average of the two middle elements (handles both odd and even-length sequences).
    median = rasp.SequenceMap(lambda x, y: (x + y) / 2, median1, median2)
    
    return median
