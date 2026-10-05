"""Choosing a control is choosing a value, so a client and the state agree."""

from __future__ import annotations

import pytest

from runtimev2.control.config import (
    DEFAULT_MODEL_HZ,
    DEFAULT_TELEOP_HZ,
    ModelControlConfig,
    TeleopControlConfig,
    describe,
    parse,
)


class TestParsing:
    def test_a_bare_kind_takes_the_default_rate(self) -> None:
        """Asking for the default should not mean naming it."""
        assert parse("teleop") == TeleopControlConfig(hz=DEFAULT_TELEOP_HZ)

    def test_a_mapping_carries_the_rate(self) -> None:
        assert parse({"kind": "model", "hz": 5}) == ModelControlConfig(hz=5.0)

    def test_a_mapping_without_a_rate_takes_the_default(self) -> None:
        assert parse({"kind": "model"}) == ModelControlConfig(hz=DEFAULT_MODEL_HZ)

    def test_null_means_nothing_drives(self) -> None:
        assert parse(None) is None

    def test_an_unknown_kind_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Unknown control 'magic'"):
            parse("magic")

    def test_a_missing_kind_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Unknown control"):
            parse({"hz": 10})

    def test_a_rate_of_zero_is_refused(self) -> None:
        """It would divide by zero in the rate loop."""
        with pytest.raises(ValueError, match="must be positive"):
            parse({"kind": "teleop", "hz": 0})

    def test_a_negative_rate_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            parse({"kind": "teleop", "hz": -1})

    def test_an_unknown_option_is_refused(self) -> None:
        """Rather than silently ignoring what a client asked for."""
        with pytest.raises(ValueError, match="Unknown options"):
            parse({"kind": "teleop", "smooth": True})


class TestDescribing:
    def test_a_config_round_trips(self) -> None:
        config = ModelControlConfig(hz=7.5)

        assert parse(describe(config)) == config

    def test_the_kind_is_included(self) -> None:
        assert describe(TeleopControlConfig())["kind"] == "teleop"
