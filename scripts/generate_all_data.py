from pathlib import Path
import sys

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.functions import generateData, getAcceptedNamesAndInput


def main():
    MAX_LENGTH = 10

    # Generate data for all programs
    for program_name in getAcceptedNamesAndInput().keys():
        print(f"Generating data for {program_name}...")

        # Generate all possible sequences
        data = generateData(name=program_name, maxSeqLength=MAX_LENGTH, exhaustive=True)

        print(f"Generated {len(data)} sequences for {program_name}")


if __name__ == "__main__":
    main()
