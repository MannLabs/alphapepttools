"""Test to ensure all public functions are documented in API.md"""

import importlib
import inspect
import re
from pathlib import Path

import pytest


def get_public_members(module_name: str) -> set[str]:
    """Get all public functions, classes, and constants from a module."""
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        print(f"Module {module_name} not found")
        print(e)

    public_members = set()

    # Check if module defines __all__, if so use that
    if hasattr(module, "__all__"):
        return set(module.__all__)

    # Otherwise, get all non-private members
    for name, obj in inspect.getmembers(module):
        # Public members don't start with underscore
        # Include functions, classes, and other callables
        if not name.startswith("_"):  # noqa: SIM102
            if inspect.isfunction(obj) or inspect.isclass(obj) or inspect.ismethod(obj) or callable(obj):
                public_members.add(name)

    return public_members


def parse_api_md(api_md_path: Path) -> dict[str, set[str]]:
    """Parse API.md to extract documented functions by module."""
    with Path.open(api_md_path) as f:
        content = f.read()

    documented_functions = {}

    # Find all autosummary blocks
    autosummary_pattern = (
        r"```\{eval-rst\}.*?.. module:: (alphatools\.\w+).*?.. autosummary::.*?:toctree: generated(.*?)```"
    )

    matches = re.findall(autosummary_pattern, content, re.DOTALL)

    for module_name, functions_block in matches:
        # Extract the short module name (pp, tl, pl, io)
        short_module = module_name.split(".")[-1]

        # Extract function names from the autosummary block
        functions = set()
        for line in functions_block.strip().split("\n"):
            stripped_line = line.strip()
            if stripped_line and not stripped_line.startswith(":"):
                # Remove module prefix (e.g., "pp.add_metadata" -> "add_metadata")
                func_name = stripped_line.split(".")[-1] if "." in stripped_line else stripped_line
                functions.add(func_name)

        documented_functions[short_module] = functions

    return documented_functions


def test_api_completeness():
    """Test that all public functions are documented in API.md"""
    modules_to_check = ["pp", "tl", "pl", "io", "metrics"]

    # Path to API.md (adjust as needed)
    api_md_path = Path("docs/api.md")

    if not api_md_path.exists():
        raise FileNotFoundError(f"API.md not found at {api_md_path}")

    # Get documented functions from API.md
    documented_functions = parse_api_md(api_md_path)

    errors = []

    for module in modules_to_check:
        module_name = f"alphatools.{module}"

        # Get actual public members from the module
        actual_members = get_public_members(module_name)

        # Get documented members from API.md
        documented_members = documented_functions.get(module, set())

        # Find missing documentation
        missing_docs = actual_members - documented_members

        # Find documented but non-existent functions
        extra_docs = documented_members - actual_members

        if missing_docs:
            errors.append(f"Module {module}: Missing documentation for: {sorted(missing_docs)}")

        if extra_docs:
            errors.append(f"Module {module}: Documented but don't exist: {sorted(extra_docs)}")

    if errors:
        error_message = "API documentation is incomplete:\n" + "\n".join(errors)
        pytest.fail(error_message)
