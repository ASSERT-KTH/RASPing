"""Generate the input-output specifications in data/ (train/val/test per program).

    python scripts/generate_data.py [--num-samples 50000] [--max-len 10] [--output-dir data/]

Random generation (duplicates removed), then a seeded 80/10/10 split. Generation is
seeded (--seed); the committed files were generated unseeded in v1 (F24) and are kept
as they are.

    python scripts/generate_data.py --program sort     # regenerate one program
"""

from pathlib import Path

import click

from rasping import data
from rasping.config import DATA_DIR, MAX_LEN, PROGRAMS


@click.command()
@click.option("--num-samples", "-n", default=50000, show_default=True)
@click.option("--max-len", "-l", default=MAX_LEN, show_default=True)
@click.option("--output-dir", "-o", default=str(DATA_DIR), show_default=True)
@click.option("--seed", default=0, show_default=True)
@click.option("--program", "only", multiple=True, help="Only these programs (default: all)")
def main(num_samples, max_len, output_dir, seed, only):
    for program in only or PROGRAMS:
        ds = data.generate(program, max_len, num_samples, remove_duplicates=True, seed=seed)
        for split, part in zip(("train", "val", "test"), data.split_train_val_test(ds)):
            data.save(program, split, part, data_dir=Path(output_dir))
        click.echo(f"{program}: {len(ds)} examples -> {output_dir}")


if __name__ == "__main__":
    main()
