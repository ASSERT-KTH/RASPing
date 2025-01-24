from generators import (
    generate_shuffle_dyck1,
    generate_shuffle_dyck2,
    generate_shuffle_dyck1_exhaustive,
    generate_shuffle_dyck2_exhaustive,
)
import numpy as np


def analyze_dyck_balance(generator, tokens, max_length, num_samples=10000):
    balanced = 0

    for _ in range(num_samples):
        input_seq, output_seq = generator(tokens, max_length)
        # Remove BOS token
        if output_seq[1:][0] == 1:  # If any output is 1, sequence is balanced
            balanced += 1

    percentage = (balanced / num_samples) * 100
    return percentage


def analyze_dyck_exhaustive(generator, tokens, max_length):
    pairs = generator(tokens, max_length)
    # All sequences from exhaustive generators are balanced
    total_balanced = len(pairs)

    # Calculate total possible sequences of each length
    total_sequences = 0
    for length in range(2, max_length + 1, 2):
        total_sequences += len(tokens) ** length

    percentage = (total_balanced / total_sequences) * 100
    return percentage, total_balanced, total_sequences


if __name__ == "__main__":
    # Test Dyck1
    dyck1_tokens = ["(", ")"]
    dyck1_balance = analyze_dyck_balance(generate_shuffle_dyck1, dyck1_tokens, 10)
    print(f"Dyck1 balanced sequences: {dyck1_balance:.2f}%")

    # Test Dyck2
    dyck2_tokens = ["(", ")", "{", "}"]
    dyck2_balance = analyze_dyck_balance(generate_shuffle_dyck2, dyck2_tokens, 10)
    print(f"Dyck2 balanced sequences: {dyck2_balance:.2f}%")

    # Test different lengths
    print("\nDyck1 balance by length:")
    for length in range(2, 11, 2):
        balance = analyze_dyck_balance(generate_shuffle_dyck1, dyck1_tokens, length)
        print(f"Length {length}: {balance:.2f}%")

    print("\nDyck2 balance by length:")
    for length in range(2, 11, 2):
        balance = analyze_dyck_balance(generate_shuffle_dyck2, dyck2_tokens, length)
        print(f"Length {length}: {balance:.2f}%")

    print("\nExhaustive Analysis:")

    # Analyze Dyck1 exhaustive
    print("\nDyck1 exhaustive by length:")
    for length in range(2, 11, 2):
        percentage, balanced, total = analyze_dyck_exhaustive(
            generate_shuffle_dyck1_exhaustive, ["(", ")"], length
        )
        print(
            f"Length {length}: {percentage:.2f}% ({balanced} out of {total} possible sequences)"
        )

    # Analyze Dyck2 exhaustive
    print("\nDyck2 exhaustive by length:")
    for length in range(2, 11, 2):
        percentage, balanced, total = analyze_dyck_exhaustive(
            generate_shuffle_dyck2_exhaustive, ["(", ")", "{", "}"], length
        )
        print(
            f"Length {length}: {percentage:.2f}% ({balanced} out of {total} possible sequences)"
        )

    print("\nComparison between random and exhaustive:")
    print("Dyck1:")
    dyck1_random = analyze_dyck_balance(generate_shuffle_dyck1, ["(", ")"], 10)
    dyck1_exhaust_percent, _, _ = analyze_dyck_exhaustive(
        generate_shuffle_dyck1_exhaustive, ["(", ")"], 10
    )
    print(f"Random: {dyck1_random:.2f}%")
    print(f"Exhaustive: {dyck1_exhaust_percent:.2f}%")

    print("\nDyck2:")
    dyck2_random = analyze_dyck_balance(
        generate_shuffle_dyck2, ["(", ")", "{", "}"], 10
    )
    dyck2_exhaust_percent, _, _ = analyze_dyck_exhaustive(
        generate_shuffle_dyck2_exhaustive, ["(", ")", "{", "}"], 10
    )
    print(f"Random: {dyck2_random:.2f}%")
    print(f"Exhaustive: {dyck2_exhaust_percent:.2f}%")
