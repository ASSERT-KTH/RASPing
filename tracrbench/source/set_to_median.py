from tracr.rasp import rasp

def make_sort() -> rasp.SOp:
    unique_tokens = rasp.SequenceMap(lambda x,y: x+0.0001*y, rasp.tokens, rasp.indices)
    smaller = rasp.Select(unique_tokens, unique_tokens, rasp.Comparison.LT).named("smaller")
    target_pos = rasp.SelectorWidth(smaller).named("target_pos")
    sel_new = rasp.Select(target_pos, rasp.indices, rasp.Comparison.EQ)
    return rasp.Aggregate(sel_new, rasp.tokens).named("sort")

def make_length() -> rasp.SOp:
    all_true_selector = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.TRUE).named("all_true_selector")
    return rasp.SelectorWidth(all_true_selector).named("length")

def make_set_to_median() -> rasp.SOp:
    sorted_sequence = make_sort()
    
    length = make_length()
    # Assuming a maximum sequence length to pre-calculate possible indices for median
    mid_index1 = rasp.Map(lambda x: (x - 1) // 2, length).named("mid_index1")
    mid_index2 = rasp.Map(lambda x: x // 2, length).named("mid_index2")
    
    # Selectors for extracting potential median values
    median_selector1 = rasp.Select(rasp.indices, mid_index1, rasp.Comparison.EQ).named("median_selector1")
    median_selector2 = rasp.Select(rasp.indices, mid_index2, rasp.Comparison.EQ).named("median_selector2")
    
    # Extracting potential median values
    potential_median1 = rasp.Aggregate(median_selector1, sorted_sequence).named("potential_median1")
    potential_median2 = rasp.Aggregate(median_selector2, sorted_sequence).named("potential_median2")
    
    # Calculating the average of the two potential medians (handles both odd and even length cases)
    median = rasp.SequenceMap(lambda x, y: (x + y) / 2, potential_median1, potential_median2).named("median")
     
    return median
