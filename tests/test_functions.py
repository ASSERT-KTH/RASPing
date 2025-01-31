import pytest
from src.functions import (
    generateData,
    generateModel,
    getAcceptedNamesAndInput,
    load_dataset,
    split_train_val_test,
    save_dataset,
)
from pathlib import Path
import json
import tempfile
import os

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


@pytest.mark.parametrize("name", ACCEPTED_NAMES)
def test_exhaustive_generation(name):
    """Test exhaustive data generation"""
    # Test with small max length to keep test runtime reasonable
    small_max_length = 3

    # Generate exhaustive data
    data = generateData(name, small_max_length, size=1000, exhaustive=True)

    # Verify all sequences are within length bounds
    for input_seq, output_seq in data:
        assert len(input_seq) <= small_max_length + 1  # +1 for BOS token
        assert len(output_seq) <= small_max_length + 1
        assert input_seq[0] == "BOS"
        assert output_seq[0] == "BOS"


@pytest.mark.parametrize("name", ACCEPTED_NAMES)
def test_exhaustive_size_limit(name):
    """Test that size parameter works with exhaustive data"""
    max_length = 3
    size = 2

    data = generateData(name, max_length, size=size, exhaustive=True)
    assert len(data) <= size, f"Generated data exceeds requested size {size}"


def create_test_data(temp_dir: Path, name: str):
    """Helper function to create test data files for a program"""
    # Generate small test datasets for each split
    data = generateData(name, MAX_SEQ_LENGTH, size=10)

    # Split the data
    train_data, val_data, test_data = split_train_val_test(data)

    # Save each split
    for split_name, split_data in [
        ("train", train_data),
        ("val", val_data),
        ("test", test_data),
    ]:
        save_dataset(temp_dir, name, split_name, split_data)


@pytest.fixture
def temp_data_dir():
    """Fixture to create a temporary directory with test data"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        for name in ACCEPTED_NAMES:
            create_test_data(temp_path, name)
        yield temp_path


@pytest.mark.parametrize("split", ["train", "val", "test"])
@pytest.mark.parametrize("name", ACCEPTED_NAMES)
def test_load_dataset(temp_data_dir, name, split):
    """Test loading data from JSONL files"""
    data = load_dataset(temp_data_dir, name, split)

    # Verify data is not empty
    assert len(data) > 0

    # Verify data format
    for input_seq, output_seq in data:
        assert isinstance(input_seq, list)
        assert isinstance(output_seq, list)
        assert len(input_seq) > 0
        assert len(output_seq) > 0
        assert input_seq[0] == "BOS"
        assert output_seq[0] == "BOS"


def test_load_dataset_invalid_path():
    """Test that load_dataset raises appropriate error for invalid paths"""
    with pytest.raises(FileNotFoundError):
        load_dataset("/nonexistent/path", "reverse", "train")


def test_load_dataset_invalid_program():
    """Test that load_dataset raises appropriate error for invalid program name"""
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(FileNotFoundError):
            load_dataset(temp_dir, "nonexistent_program", "train")


def test_load_dataset_invalid_split():
    """Test that load_dataset raises appropriate error for invalid split name"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        # Create test data for one program
        create_test_data(temp_path, "reverse")
        with pytest.raises(FileNotFoundError):
            load_dataset(temp_path, "reverse", "invalid_split")


def test_split_train_val_test():
    """Test that data splitting works correctly"""
    # Create test data
    data = [(i, i) for i in range(100)]  # 100 dummy samples

    # Test with default split ratios
    train, val, test = split_train_val_test(data)
    assert len(train) == 80  # 80% of 100
    assert len(val) == 10  # 10% of 100
    assert len(test) == 10  # remaining 10%

    # Test with custom split ratios
    train, val, test = split_train_val_test(data, train_pct=0.6, val_pct=0.2)
    assert len(train) == 60  # 60% of 100
    assert len(val) == 20  # 20% of 100
    assert len(test) == 20  # remaining 20%

    # Verify no data is lost or duplicated
    all_data = train + val + test
    assert len(all_data) == len(data)
    assert len(set(str(x) for x in all_data)) == len(data)  # Check uniqueness


def test_save_dataset():
    """Test saving dataset to JSONL file"""
    # Create test data and temporary directory
    test_data = [
        (["BOS", "a", "b"], ["BOS", "b", "a"]),
        (["BOS", "1", "2"], ["BOS", "2", "1"]),
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save and reload the data
        tmp_path = Path(tmp_dir)
        save_dataset(tmp_path, "test", "train", test_data)

        # Test file exists and content is correct
        loaded_data = load_dataset(tmp_path, "test", "train")
        assert len(loaded_data) == len(test_data)
        assert all(a == b for a, b in zip(loaded_data, test_data))


if __name__ == "__main__":
    pytest.main([__file__])
