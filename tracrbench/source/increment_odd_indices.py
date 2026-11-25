from tracr.rasp import rasp

def make_increment_odd_indices() -> rasp.SOp:
    # Marks odd indices with 1 and even indices with 0
    odd_index_marker = rasp.Map(lambda x: x % 2, rasp.indices).named("odd_index_marker")

    # Increment elements at odd indices by 1
    incremented_elements = rasp.SequenceMap(
        lambda elem, mark: elem + mark, rasp.tokens, odd_index_marker
    ).named("incremented_elements")

    return incremented_elements
