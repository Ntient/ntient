import pytest
import os


@pytest.fixture
def cli_context():
    os.environ["NTIENT_TOKEN"] = "test_token"
    os.environ["NTIENT_HOST"] = "ntient.ai/api"
