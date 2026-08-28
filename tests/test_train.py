"""Tests for configuration resolution, environment overrides and early stopping.

These cover the parts of `train.py` that every Docker and Kubernetes run depends
on but that no test previously touched. Both areas had been verified only by
reading log output from real runs, which proves a behaviour happened once — not
that it will keep happening.
"""

from __future__ import annotations

import pytest
import yaml

from train import (
    ENV_OVERRIDES,
    EarlyStopping,
    apply_env_overrides,
    load_config,
    resolve_config_path,
)

ALL_ENV_VARS = [name for name, _, _, _ in ENV_OVERRIDES]


@pytest.fixture(autouse=True)
def clear_override_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test from a clean environment.

    Without this, a variable exported in the developer's shell — exactly the
    ones used to shorten demo runs — would leak in and change the results.
    """
    for name in [*ALL_ENV_VARS, "TRAINING_CONFIG_PATH"]:
        monkeypatch.delenv(name, raising=False)


def baseline_config() -> dict:
    """A config matching the committed defaults and the Kubernetes ConfigMap."""
    return {
        "model": {"architecture": "resnet18", "num_classes": 10},
        "training": {
            "epochs": 10,
            "batch_size": 64,
            "learning_rate": 0.001,
            "early_stopping_patience": 3,
            "num_workers": 2,
            "seed": 42,
        },
        "data": {"dataset": "cifar10", "data_dir": "/app/data", "subset_fraction": 1.0},
        "output": {"checkpoint_dir": "/app/checkpoints", "model_name": "classifier_v1.pt"},
    }


# ---------------------------------------------------------------------------
# Environment overrides
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("env_name", "raw", "section", "key", "expected"),
    [
        ("MAX_EPOCHS", "2", "training", "epochs", 2),
        ("BATCH_SIZE", "128", "training", "batch_size", 128),
        ("LEARNING_RATE", "0.01", "training", "learning_rate", 0.01),
        ("NUM_WORKERS", "0", "training", "num_workers", 0),
        ("SUBSET_FRACTION", "0.05", "data", "subset_fraction", 0.05),
        ("DATA_DIR", "/tmp/data", "data", "data_dir", "/tmp/data"),
        ("CHECKPOINT_DIR", "/tmp/ckpt", "output", "checkpoint_dir", "/tmp/ckpt"),
    ],
)
def test_each_override_reaches_its_config_key(
    monkeypatch: pytest.MonkeyPatch,
    env_name: str,
    raw: str,
    section: str,
    key: str,
    expected: object,
) -> None:
    """Every documented variable lands in the right place with the right type."""
    monkeypatch.setenv(env_name, raw)

    result = apply_env_overrides(baseline_config())

    assert result[section][key] == expected
    assert type(result[section][key]) is type(expected)


def test_every_documented_variable_is_covered() -> None:
    """Guards against a new override being added without a test.

    If someone extends ENV_OVERRIDES, this fails until the parametrised list
    above is extended too.
    """
    tested = {
        "MAX_EPOCHS",
        "BATCH_SIZE",
        "LEARNING_RATE",
        "NUM_WORKERS",
        "SUBSET_FRACTION",
        "DATA_DIR",
        "CHECKPOINT_DIR",
    }
    assert set(ALL_ENV_VARS) == tested


def test_unset_variables_leave_the_config_untouched() -> None:
    """With no environment set, the YAML values survive verbatim."""
    assert apply_env_overrides(baseline_config()) == baseline_config()


def test_malformed_value_is_ignored_not_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bad value must not crash a training run that is otherwise fine.

    A Kubernetes Job that dies on a typo in one environment variable is worse
    than one that logs the problem and uses the configured default.
    """
    monkeypatch.setenv("MAX_EPOCHS", "not-a-number")
    monkeypatch.setenv("BATCH_SIZE", "128")

    result = apply_env_overrides(baseline_config())

    assert result["training"]["epochs"] == 10, "bad value should fall back to YAML"
    assert result["training"]["batch_size"] == 128, "good values still apply"


def test_malformed_value_is_reported(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Ignoring a bad value silently would be worse than crashing."""
    monkeypatch.setenv("SUBSET_FRACTION", "half")

    apply_env_overrides(baseline_config())

    events = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert any("invalid_env_override" in line and "SUBSET_FRACTION" in line for line in events)


def test_empty_string_is_treated_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """An env var set to "" is how Kubernetes represents an unset optional value."""
    monkeypatch.setenv("MAX_EPOCHS", "")

    assert apply_env_overrides(baseline_config())["training"]["epochs"] == 10


def test_override_creates_a_missing_section(monkeypatch: pytest.MonkeyPatch) -> None:
    """A config lacking the target section should gain it rather than raise."""
    monkeypatch.setenv("CHECKPOINT_DIR", "/tmp/ckpt")

    result = apply_env_overrides({"model": {"architecture": "resnet18"}})

    assert result["output"]["checkpoint_dir"] == "/tmp/ckpt"


def test_float_override_accepts_an_integer_literal(monkeypatch: pytest.MonkeyPatch) -> None:
    """`SUBSET_FRACTION=1` should mean 1.0, not fail."""
    monkeypatch.setenv("SUBSET_FRACTION", "1")

    value = apply_env_overrides(baseline_config())["data"]["subset_fraction"]

    assert value == 1.0
    assert isinstance(value, float)


def test_applied_overrides_are_logged(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The run log must record what was overridden, or results are unexplainable."""
    monkeypatch.setenv("MAX_EPOCHS", "2")
    monkeypatch.setenv("SUBSET_FRACTION", "0.05")

    apply_env_overrides(baseline_config())

    out = capsys.readouterr().out
    assert "env_overrides_applied" in out
    assert "MAX_EPOCHS" in out and "SUBSET_FRACTION" in out


# ---------------------------------------------------------------------------
# Configuration resolution
# ---------------------------------------------------------------------------


def test_explicit_config_path_wins(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """TRAINING_CONFIG_PATH takes precedence over the search order."""
    config_file = tmp_path / "custom.yaml"
    config_file.write_text(yaml.safe_dump(baseline_config()))
    monkeypatch.setenv("TRAINING_CONFIG_PATH", str(config_file))

    assert resolve_config_path() == config_file


def test_missing_explicit_path_fails_loudly(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Pointing at a nonexistent file must not silently fall back.

    Falling through to a different config would train on settings the operator
    did not ask for, and the logs would look entirely normal.
    """
    monkeypatch.setenv("TRAINING_CONFIG_PATH", str(tmp_path / "absent.yaml"))

    with pytest.raises(FileNotFoundError, match="TRAINING_CONFIG_PATH"):
        resolve_config_path()


def test_falls_back_to_the_repository_config() -> None:
    """With no override set, the committed config is found."""
    assert resolve_config_path().name == "training_config.yaml"


def test_config_roundtrips(tmp_path) -> None:
    config_file = tmp_path / "c.yaml"
    config_file.write_text(yaml.safe_dump(baseline_config()))

    assert load_config(config_file) == baseline_config()


def test_non_mapping_config_is_rejected(tmp_path) -> None:
    """A YAML list or scalar would fail later with a confusing KeyError."""
    config_file = tmp_path / "bad.yaml"
    config_file.write_text("- just\n- a\n- list\n")

    with pytest.raises(ValueError, match="did not parse to a mapping"):
        load_config(config_file)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


def test_first_epoch_always_improves() -> None:
    assert EarlyStopping(patience=3).update(1, val_loss=2.5, val_accuracy=0.1) is True


def test_lower_loss_improves_and_higher_does_not() -> None:
    stopper = EarlyStopping(patience=3)
    stopper.update(1, 1.0, 0.5)

    assert stopper.update(2, 0.9, 0.6) is True
    assert stopper.update(3, 1.1, 0.4) is False


def test_equal_loss_is_not_an_improvement() -> None:
    """The comparison is strict, so a plateau counts against patience."""
    stopper = EarlyStopping(patience=3)
    stopper.update(1, 1.0, 0.5)

    assert stopper.update(2, 1.0, 0.5) is False
    assert stopper.counter == 1


def test_stops_after_patience_consecutive_failures() -> None:
    stopper = EarlyStopping(patience=3)
    stopper.update(1, 1.0, 0.5)

    for epoch, loss in [(2, 1.1), (3, 1.2)]:
        stopper.update(epoch, loss, 0.4)
        assert not stopper.should_stop

    stopper.update(4, 1.3, 0.4)
    assert stopper.should_stop


def test_an_improvement_resets_the_counter() -> None:
    """Patience counts consecutive failures, not cumulative ones.

    This is the rule that keeps a single bad epoch from ending a good run.
    """
    stopper = EarlyStopping(patience=2)
    stopper.update(1, 1.0, 0.5)
    stopper.update(2, 1.1, 0.4)
    assert stopper.counter == 1

    stopper.update(3, 0.8, 0.7)
    assert stopper.counter == 0
    assert not stopper.should_stop

    stopper.update(4, 0.9, 0.6)
    assert stopper.counter == 1
    assert not stopper.should_stop, "two non-consecutive failures must not stop training"


def test_reproduces_the_full_run_sequence() -> None:
    """The observed validation losses from the real 10-epoch run.

    Epoch 8 regressed and epoch 9 recovered. With patience 3 the run must
    complete all ten epochs and report epoch 10 as the best — which is what
    happened, ending at 0.8718 validation accuracy.
    """
    observed = [
        (1, 1.0745, 0.6227),
        (2, 0.8083, 0.7209),
        (3, 0.7432, 0.7488),
        (4, 0.6008, 0.7937),
        (5, 0.5295, 0.8198),
        (6, 0.5130, 0.8284),
        (7, 0.4682, 0.8425),
        (8, 0.5041, 0.8356),  # the regression
        (9, 0.4598, 0.8540),  # recovery resets the counter
        (10, 0.3947, 0.8718),
    ]
    stopper = EarlyStopping(patience=3)
    improved_at = [epoch for epoch, loss, acc in observed if stopper.update(epoch, loss, acc)]

    assert not stopper.should_stop, "the run must not have stopped early"
    assert improved_at == [1, 2, 3, 4, 5, 6, 7, 9, 10], "epoch 8 is the only regression"
    assert stopper.best_epoch == 10
    assert stopper.best_loss == pytest.approx(0.3947)
    assert stopper.best_accuracy == pytest.approx(0.8718)


def test_best_metrics_track_the_best_not_the_last() -> None:
    """The checkpoint on disk is the best model, so the reported metrics must match."""
    stopper = EarlyStopping(patience=5)
    stopper.update(1, 1.0, 0.50)
    stopper.update(2, 0.5, 0.80)
    stopper.update(3, 0.9, 0.55)

    assert stopper.best_epoch == 2
    assert stopper.best_loss == pytest.approx(0.5)
    assert stopper.best_accuracy == pytest.approx(0.80)


def test_patience_of_one_stops_on_the_first_failure() -> None:
    stopper = EarlyStopping(patience=1)
    stopper.update(1, 1.0, 0.5)

    stopper.update(2, 1.1, 0.4)

    assert stopper.should_stop


@pytest.mark.parametrize("patience", [0, -1])
def test_invalid_patience_is_rejected(patience: int) -> None:
    """Patience 0 would stop before any epoch could fail."""
    with pytest.raises(ValueError, match="at least 1"):
        EarlyStopping(patience=patience)
