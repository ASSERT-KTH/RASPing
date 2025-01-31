from pathlib import Path
import sys
import click

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.functions import (
    generateData,
    getAcceptedNamesAndInput,
    split_train_val_test,
    save_dataset,
)


@click.command()
@click.option(
    "--num-samples",
    "-n",
    default=50000,
    help="Number of samples to generate per program",
)
@click.option("--max-len", "-l", default=10, help="Maximum sequence length")
@click.option(
    "--output-dir",
    "-o",
    default="data/",
    help="Output directory for the generated datasets",
)
def main(num_samples: int, max_len: int, output_dir: str):
    """Generate datasets for all supported programs with train/val/test splits."""
    output_path = Path(output_dir)

    # Generate data for all programs
    for program_name in getAcceptedNamesAndInput().keys():
        click.echo(f"Generating {num_samples} samples for {program_name}...")

        # Use exhaustive generation for shuffle_dyck programs
        use_exhaustive = program_name.startswith("shuffle_dyck")

        # Generate samples with no duplicates
        data = generateData(
            name=program_name,
            maxSeqLength=max_len,
            size=num_samples,
            removeDuplicates=True,
            exhaustive=use_exhaustive,
        )

        click.echo(f"Generated {len(data)} sequences for {program_name}")

        # Split the data
        train_data, val_data, test_data = split_train_val_test(data)

        # Save each split
        for split_name, split_data in [
            ("train", train_data),
            ("val", val_data),
            ("test", test_data),
        ]:
            save_dataset(output_path, program_name, split_name, split_data)

        click.echo(f"Saved splits to {output_path}")


if __name__ == "__main__":
    main()
