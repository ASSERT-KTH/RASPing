import pytest
from load_mutations import load_buggy_models

from src.model import Model


@pytest.mark.parametrize(
    "program_name,job_id",
    [
        ("sort", "4795828b71974ec8829297c301a19dfe"),
        ("reverse", "2b6f62a96b1d411a81185c47fcf323ab"),
        ("hist", "2a722d5d5449441fad93f98c73c633a2"),
        ("most_freq", "d6de3b3af9fe403faeac2b7b7f0b5f85"),
        ("shuffle_dyck", "28357cf95fbd4389a311073d185756b8"),
        ("shuffle_dyck2", "95b38e6e8ede42268ff62e85f987dfa7"),
    ],
)
def test_load_program_mutations(program_name, job_id):
    """Test loading mutations for each supported program."""
    models = load_buggy_models(max_length=10, program_name=program_name, job_id=job_id)
    assert isinstance(models, dict)
    assert len(models) == 1, f"Expected exactly one mutation for program {program_name}"
    # Verify all loaded models are for the requested program
    for mutation_id, model in models.items():
        assert isinstance(model, Model)


@pytest.mark.parametrize(
    "program_name",
    ["sort", "reverse", "hist", "most_freq", "shuffle_dyck", "shuffle_dyck2"],
)
@pytest.mark.slow
def test_load_all_program_mutations(program_name):
    """Test loading all mutations for each supported program."""
    models = load_buggy_models(max_length=10, program_name=program_name)
    assert isinstance(models, dict)
    assert len(models) > 0, f"Expected at least one mutation for program {program_name}"
    # Verify all loaded models are for the requested program
    for mutation_id, model in models.items():
        assert isinstance(model, Model)
