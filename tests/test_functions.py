import pytest
from ..src.functions import generateData, generateModel, getAcceptedNamesAndInput

ACCEPTED_NAMES = list(getAcceptedNamesAndInput().keys())
TEST_SIZE = 50_000
SMALL_TEST_SIZE = 100
MAX_SEQ_LENGTH = 10


@pytest.fixture
def program_names():
    return ACCEPTED_NAMES


@pytest.mark.parametrize("name", ACCEPTED_NAMES)
def test_data_generation(name):
    """Test that we can generate large datasets for each program"""
    data = generateData(name, MAX_SEQ_LENGTH, TEST_SIZE)
    assert len(data) == TEST_SIZE

    # Test that sequences are within length bounds
    for input_seq, output_seq in data:
        assert len(input_seq) <= MAX_SEQ_LENGTH + 1  # +1 for BOS token
        assert len(output_seq) <= MAX_SEQ_LENGTH + 1  # +1 for BOS token
        assert len(input_seq) == len(output_seq)
        assert input_seq[0] == "BOS"
        assert output_seq[0] == "BOS"


@pytest.mark.parametrize("name", ACCEPTED_NAMES)
def test_model_correctness(name):
    """Test that models produce correct outputs for each program"""
    # Generate a small test set for quick validation
    data = generateData(name, MAX_SEQ_LENGTH, SMALL_TEST_SIZE)
    model = generateModel(name, MAX_SEQ_LENGTH)

    # Use evaluateModel with useAssert=True to check correctness
    # This will raise an AssertionError if any output is incorrect
    accuracy = model.evaluateModel(
        data, doPrint=False, outputArray=False, useAssert=True
    )
    assert accuracy == 1.0


@pytest.mark.parametrize("name", ACCEPTED_NAMES)
def test_model_compilation(name):
    """Test that we can compile models for each program"""
    model = generateModel(name, MAX_SEQ_LENGTH)
    assert model is not None

    # Test basic model properties
    assert model.name == name
    assert model.seqLength == MAX_SEQ_LENGTH
    assert model.model is not None


def test_data_uniqueness():
    """Test that removeDuplicates option works correctly"""
    name = ACCEPTED_NAMES[0]  # Test with first program
    data_unique = generateData(
        name, MAX_SEQ_LENGTH, SMALL_TEST_SIZE, removeDuplicates=True
    )

    # Convert data to hashable format for set comparison
    data_unique_set = {(tuple(x), tuple(y)) for x, y in data_unique}

    # Check that unique data has no duplicates
    assert len(data_unique) == len(data_unique_set)
    # Check that unique data might be smaller than requested size due to duplicates
    assert len(data_unique) <= SMALL_TEST_SIZE


if __name__ == "__main__":
    pytest.main([__file__])
