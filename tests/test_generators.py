from ..src.generators import (
    generate_reverse_exhaustive,
    generate_hist_exhaustive,
    generate_sort_exhaustive,
    generate_most_freq_exhaustive,
    generate_shuffle_dyck1_exhaustive,
    generate_shuffle_dyck2_exhaustive,
)

from typing import Counter


def test_generate_reverse_exhaustive():
    tokens = ["a", "b", "c"]
    max_length = 3

    pairs = generate_reverse_exhaustive(tokens, max_length)

    # Calculate expected total number of sequences
    # For n tokens and max_length l, total = n^2 + n^3 + ... + n^l
    expected_total = sum(len(tokens) ** i for i in range(2, max_length + 1))
    assert len(pairs) == expected_total

    # Check properties for all sequences
    for input_seq, output_seq in pairs:
        # Check BOS tokens
        assert input_seq[0] == "BOS"
        assert output_seq[0] == "BOS"

        # Check reverse relationship
        assert output_seq[1:] == input_seq[1:][::-1]

        # Check lengths
        assert len(input_seq) == len(output_seq)
        assert 3 <= len(input_seq) <= max_length + 1  # +1 for BOS token

        # Check all tokens are from accepted set
        assert all(t in tokens or t == "BOS" for t in input_seq)
        assert all(t in tokens or t == "BOS" for t in output_seq)


def test_generate_hist_exhaustive():
    tokens = ["a", "b", "c"]
    max_length = 3

    pairs = generate_hist_exhaustive(tokens, max_length)

    expected_total = sum(len(tokens) ** i for i in range(2, max_length + 1))
    assert len(pairs) == expected_total

    for input_seq, output_seq in pairs:
        assert input_seq[0] == "BOS"
        assert output_seq[0] == "BOS"

        # Check histogram relationship
        counts = Counter(input_seq[1:])
        assert all(
            output_seq[i] == counts[input_seq[i]] for i in range(1, len(input_seq))
        )

        assert len(input_seq) == len(output_seq)
        assert 3 <= len(input_seq) <= max_length + 1


def test_generate_sort_exhaustive():
    tokens = [1, 2, 3]
    max_length = 3

    pairs = generate_sort_exhaustive(tokens, max_length)

    expected_total = sum(len(tokens) ** i for i in range(2, max_length + 1))
    assert len(pairs) == expected_total

    for input_seq, output_seq in pairs:
        assert input_seq[0] == "BOS"
        assert output_seq[0] == "BOS"

        # Check sort relationship
        assert output_seq[1:] == sorted(input_seq[1:])

        assert len(input_seq) == len(output_seq)
        assert 3 <= len(input_seq) <= max_length + 1


def test_generate_most_freq_exhaustive():
    tokens = [1, 2, 3]
    max_length = 3

    pairs = generate_most_freq_exhaustive(tokens, max_length)

    expected_total = sum(len(tokens) ** i for i in range(2, max_length + 1))
    assert len(pairs) == expected_total

    for input_seq, output_seq in pairs:
        assert input_seq[0] == "BOS"
        assert output_seq[0] == "BOS"

        # Check frequency-based sorting
        counts = Counter(input_seq[1:])
        sorted_tokens = sorted(input_seq[1:], key=lambda x: (-counts[x], x))
        assert output_seq[1:] == sorted_tokens

        assert len(input_seq) == len(output_seq)
        assert 3 <= len(input_seq) <= max_length + 1


def test_generate_shuffle_dyck1_exhaustive():
    max_length = 4
    pairs = generate_shuffle_dyck1_exhaustive(["(", ")"], max_length)

    # Calculate number of sequences - should include all possible combinations
    total_sequences = sum(2**i for i in range(2, max_length + 1))
    assert len(pairs) == total_sequences

    for input_seq, output_seq in pairs:
        assert input_seq[0] == "BOS"
        assert output_seq[0] == "BOS"

        # Check if sequence is balanced
        sequence = input_seq[1:]
        balance = 0
        valid = True
        for t in sequence:
            balance += 1 if t == "(" else -1
            if balance < 0:
                valid = False
                break

        # Output should be 1 only if sequence is balanced (valid and final balance is 0)
        expected_output = 1 if valid and balance == 0 else 0
        assert all(x == expected_output for x in output_seq[1:])

        assert 2 <= len(sequence) <= max_length


def test_generate_shuffle_dyck2_exhaustive():
    max_length = 4
    pairs = generate_shuffle_dyck2_exhaustive(["(", ")", "{", "}"], max_length)

    # Calculate number of sequences - should include all possible combinations
    total_sequences = sum(4**i for i in range(2, max_length + 1))
    assert len(pairs) == total_sequences

    for input_seq, output_seq in pairs:
        assert input_seq[0] == "BOS"
        assert output_seq[0] == "BOS"

        # Check if sequence is balanced for both bracket types
        sequence = input_seq[1:]
        balance1 = balance2 = 0
        valid = True
        for t in sequence:
            if t == "(":
                balance1 += 1
            elif t == ")":
                balance1 -= 1
            elif t == "{":
                balance2 += 1
            else:  # t == '}'
                balance2 -= 1

            if balance1 < 0 or balance2 < 0:
                valid = False
                break

        # Output should be 1 only if sequence is balanced (valid and both balances are 0)
        expected_output = 1 if valid and balance1 == 0 and balance2 == 0 else 0
        assert all(x == expected_output for x in output_seq[1:])

        assert 2 <= len(sequence) <= max_length
