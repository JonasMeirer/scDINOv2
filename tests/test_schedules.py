"""Tests for the training schedules (teacher temperature, weight decay, momentum)."""

import pytest

from scdino.models.lightning.utils import cosine_schedule, linear_warmup_schedule


class TestLinearWarmupSchedule:
    def test_endpoints(self):
        assert linear_warmup_schedule(0, 100, 0.04, 0.07) == pytest.approx(0.04)
        assert linear_warmup_schedule(100, 100, 0.04, 0.07) == pytest.approx(0.07)

    def test_midpoint_is_halfway(self):
        assert linear_warmup_schedule(50, 100, 0.04, 0.08) == pytest.approx(0.06)

    def test_clamps_after_warmup(self):
        assert linear_warmup_schedule(10_000, 100, 0.04, 0.07) == pytest.approx(0.07)

    def test_monotonically_increasing(self):
        values = [linear_warmup_schedule(s, 50, 0.04, 0.07) for s in range(60)]
        assert all(b >= a for a, b in zip(values, values[1:]))

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"step": -1, "warmup_steps": 10, "start_value": 0.04, "end_value": 0.07},
            {"step": 0, "warmup_steps": -1, "start_value": 0.04, "end_value": 0.07},
            {"step": 0, "warmup_steps": 10, "start_value": -0.1, "end_value": 0.07},
            {"step": 0, "warmup_steps": 10, "start_value": 0.04, "end_value": 0.0},
            {"step": 0, "warmup_steps": 10, "start_value": 0.09, "end_value": 0.07},
        ],
        ids=[
            "negative-step",
            "negative-warmup",
            "negative-start",
            "zero-end",
            "start>end",
        ],
    )
    def test_rejects_invalid_arguments(self, kwargs):
        with pytest.raises(ValueError):
            linear_warmup_schedule(**kwargs)


class TestCosineSchedule:
    def test_endpoints(self):
        assert cosine_schedule(0, 100, 0.9, 1.0) == pytest.approx(0.9)
        assert cosine_schedule(99, 100, 0.9, 1.0) == pytest.approx(1.0)

    def test_monotone_between_endpoints(self):
        values = [cosine_schedule(s, 100, 0.992, 1.0) for s in range(100)]
        assert all(b >= a for a, b in zip(values, values[1:]))
        assert min(values) >= 0.992
        assert max(values) <= 1.0

    def test_decaying_direction(self):
        """start > end must decay rather than grow."""
        values = [cosine_schedule(s, 100, 1.0, 0.1) for s in range(100)]
        assert all(b <= a for a, b in zip(values, values[1:]))

    def test_degenerate_max_steps_does_not_divide_by_zero(self):
        assert cosine_schedule(0, 1, 0.9, 1.0) == pytest.approx(1.0)
        assert cosine_schedule(0, 0, 0.9, 1.0) == pytest.approx(1.0)

    def test_periodic_mode_cycles(self):
        a = cosine_schedule(0, 100, 0.0, 1.0, period=20)
        b = cosine_schedule(20, 100, 0.0, 1.0, period=20)
        assert a == pytest.approx(b)

    def test_rejects_invalid_arguments(self):
        with pytest.raises(ValueError):
            cosine_schedule(-1, 100, 0.9, 1.0)
        with pytest.raises(ValueError):
            cosine_schedule(0, -1, 0.9, 1.0)
        with pytest.raises(ValueError):
            cosine_schedule(0, 100, 0.9, 1.0, period=0)

    def test_warns_when_step_exceeds_max_steps(self):
        with pytest.warns(RuntimeWarning):
            cosine_schedule(200, 100, 0.9, 1.0)
