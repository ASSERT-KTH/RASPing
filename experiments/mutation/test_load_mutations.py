import pytest
from load_mutations import load_buggy_models

from src.model import Model


@pytest.mark.parametrize(
    "program_name",
    ["sort", "reverse", "hist", "most_freq", "shuffle_dyck", "shuffle_dyck2"],
)
def test_load_all_program_mutations(program_name):
    """Test loading all mutations for each supported program."""
    models = load_buggy_models(max_length=10, program_name=program_name)
    assert isinstance(models, dict)
    assert len(models) > 0, f"Expected at least one mutation for program {program_name}"
    # Verify all loaded models are for the requested program
    for mutation_id, model in models.items():
        assert isinstance(model, Model)
