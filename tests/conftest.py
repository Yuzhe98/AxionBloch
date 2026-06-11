"""Shared pytest configuration and fixtures for the axionbloch test suite."""

import pytest


def pytest_configure(config):
    import matplotlib

    if not getattr(config.option, "show_plots", False):
        matplotlib.use("Agg")  # non-interactive; prevents Tk init in headless runs


def pytest_addoption(parser):
    parser.addoption(
        "--show-plots",
        action="store_true",
        default=False,
        help="Display diagnostic matplotlib figures during test runs.",
    )


@pytest.fixture
def show_plots(request) -> bool:
    """Return True when --show-plots is passed on the command line."""
    return request.config.getoption("--show-plots")
