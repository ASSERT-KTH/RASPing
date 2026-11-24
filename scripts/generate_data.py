"""Generate the input-output specifications in data/ (train/val/test per program).

    python scripts/generate_data.py [--num-samples 50000] [--max-len 10] [--output-dir data/]

Random generation for the sequence-to-sequence programs (duplicates removed),
exhaustive enumeration (truncated to --num-samples) for the Dyck programs, then a
seeded 80/10/10 split. NOTE: as in v1, the random generators use NumPy's unseeded
global RNG (F24) and exhaustive enumeration is truncated before shuffling (F1).
"""

from pathlib import Path

import click

from rasping import data
from rasping.config import DATA_DIR, MAX_LEN, PROGRAMS


@click.command()
@click.option("--num-samples", "-n", default=50000, show_default=True)
@click.option("--max-len", "-l", default=MAX_LEN, show_default=True)
@click.option("--output-dir", "-o", default=str(DATA_DIR), show_default=True)
def main(num_samples, max_len, output_dir):
    for program in PROGRAMS:
        exhaustive = program.startswith("shuffle_dyck")
        ds = data.generate(program, max_len, num_samples, remove_duplicates=True, exhaustive=exhaustive)
        for split, part in zip(("train", "val", "test"), data.split_train_val_test(ds)):
            data.save(program, split, part, data_dir=Path(output_dir))
        click.echo(f"{program}: {len(ds)} examples -> {output_dir}")


if __name__ == "__main__":
    main()
