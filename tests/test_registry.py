"""Model registry lifecycle: register -> promote -> freeze -> retire rules."""

import pytest

from core import registry


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    from database.db import init_db, session_scope
    from database.models import ModelVersion
    init_db()
    with session_scope() as s:
        s.query(ModelVersion).delete()
    monkeypatch.setattr(registry, "REGISTRY_DIR", tmp_path)
    monkeypatch.setattr(registry, "REGISTRY_INDEX", tmp_path / "registry.json")
    yield tmp_path


def _fake_weights(tmp_path, name="best.pt"):
    p = tmp_path / name
    p.write_bytes(b"fake-weights")
    return p


def test_register_and_promote(tmp_path):
    registry.register_model("v1.0.0", str(_fake_weights(tmp_path)))
    assert registry.get_active_model() is None  # candidate != deployed
    registry.promote_model("v1.0.0")
    version, weights = registry.get_active_model()
    assert version == "v1.0.0" and weights.exists()


def test_cannot_retire_active(tmp_path):
    registry.register_model("v1.0.0", str(_fake_weights(tmp_path)))
    registry.promote_model("v1.0.0")
    with pytest.raises(ValueError):
        registry.retire_model("v1.0.0")


def test_promote_unknown_version_fails():
    with pytest.raises(KeyError):
        registry.promote_model("v9.9.9")
