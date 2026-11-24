def pytest_addoption(parser):
    parser.addoption(
        "--golden-exact", action="store_true", default=False,
        help="compare whole golden training runs to 1e-7 (only reproducible on the machine and "
             "thread count that recorded tests/golden/golden_v1.json)",
    )
