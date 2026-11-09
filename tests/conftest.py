"""
Pytest shared fixtures configuration for LoanWatch test suite.
"""

from pathlib import Path
import tempfile
import pytest

from src.db import init_db


@pytest.fixture
def temp_db():
    """Provides an isolated temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_path = Path(f.name)
    init_db(temp_path)
    yield temp_path
    if temp_path.exists():
        temp_path.unlink()
