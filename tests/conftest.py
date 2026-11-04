import os

import pytest

os.environ.setdefault("GGUF_HACK_NO_PROGRESS", "1")
os.environ.setdefault("GGUF_HACK_LOGLEVEL", "WARNING")


@pytest.fixture(scope="session")
def model_template(tmp_path_factory):
    from gguf_weight_hacker.testmodel import build_tiny_model
    return build_tiny_model(str(tmp_path_factory.mktemp("tpl") / "tiny.gguf"), seed=7)


@pytest.fixture
def model(model_template, tmp_path):
    import shutil
    p = tmp_path / "tiny.gguf"
    shutil.copy(model_template, p)
    return str(p)
