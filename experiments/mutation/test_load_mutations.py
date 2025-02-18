import pytest
from load_mutations import load_buggy_models

from src.model import Model


@pytest.mark.parametrize(
    "program_name,job_id",
    [
        ("sort", "f35ba7e838874bba8335cd9ca5db2aa7"),
        ("reverse", "9db50f10314547858e52a5aff4bc2be4"),
        ("hist", "45728e11fb1043829d4057c016b549b9"),
        ("most_freq", "14ac16fe5f49412aa1ee30461b5769a0"),
        ("shuffle_dyck", "8940b2d2299f45c08b474d151c6d760b"),
        ("shuffle_dyck2", "739a764b78784fc3b4b6f80006eac399"),
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
def test_load_all_program_mutations(program_name):
    """Test loading all mutations for each supported program."""
    models = load_buggy_models(max_length=10, program_name=program_name)
    assert isinstance(models, dict)
    assert len(models) > 0, f"Expected at least one mutation for program {program_name}"
    # Verify all loaded models are for the requested program
    for mutation_id, model in models.items():
        assert isinstance(model, Model)
