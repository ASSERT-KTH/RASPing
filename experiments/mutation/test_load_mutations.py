import pytest
from load_mutations import load_buggy_models

from src.model import Model


@pytest.mark.parametrize(
    "program_name,job_id",
    [
        ("sort", "5f1e78eb029f49b693b2653592dd7dea"),
        ("reverse", "2564f0095e7d430d88cb958a8e948972"),
        ("hist", "4877157928854d51aa44ea18a425e1cb"),
        ("most_freq", "522c0475818042c4bead3e467f62aefb"),
        ("shuffle_dyck", "0025d214e8ad403f9771a909c9679d65"),
        ("shuffle_dyck2", "6bb5a25d429243a9b199efb4d54cc1f0"),
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
