from setuptools import setup

setup(
    name="tracr_operators",
    version="0.1.0",
    entry_points={
        "cosmic_ray.operator_providers": ["operators = operators.provider:Provider"]
    },
)
