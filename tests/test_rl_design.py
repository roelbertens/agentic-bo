"""Unit tests for the reporting helpers in ``rl_design.py``.

Every number in the results tables passes through these: ``_smooth`` (planner-curve
smoothing), ``plateau`` (the value a jittery method converges to) and ``reaches_at``
(the episode a method first reaches its result). They are tested here on synthetic
curves with known answers, so a reporting change cannot silently reshape the story.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rl_design import _smooth, plateau, reaches_at


def test_smooth_preserves_length_and_constants():
    v = np.full(50, 3.0)
    s = _smooth(v)
    assert s.shape == v.shape and np.allclose(s, 3.0)


def test_smooth_steadies_a_jittery_plateau_without_shifting_it():
    rng = np.random.default_rng(0)
    v = 5.0 + rng.normal(0.0, 1.0, size=200)
    s = _smooth(v)
    assert np.std(s[50:]) < np.std(v[50:]) / 2      # jitter reduced ...
    assert abs(np.mean(s[-20:]) - 5.0) < 0.5        # ... around the same level


def test_plateau_is_the_mean_of_the_tail():
    v = np.concatenate([np.zeros(30), np.full(10, 4.0)])
    assert plateau(v) == 4.0                        # last quarter sits on the plateau


def test_reaches_at_measures_the_full_climb_to_the_optimum():
    # a method whose result is the optimum is measured against the optimum's
    # 99%-of-span threshold — the slow last percent of the climb included
    samples = np.arange(1, 101)
    values = np.linspace(0.0, 10.0, 100)
    assert reaches_at(samples, values, 10.0, optimum=10.0, rnd=0.0) == 100


def test_reaches_at_uses_the_own_plateau_when_capped_below():
    # a method that plateaus below the optimum is measured against its own plateau,
    # not the optimum it never reaches
    samples = np.arange(1, 51)
    values = np.concatenate([np.linspace(0.0, 7.0, 20), np.full(30, 7.0)])
    assert reaches_at(samples, values, 7.0, optimum=10.0, rnd=0.0) == 20


def test_reaches_at_falls_back_to_the_last_sample():
    # a curve still climbing at its budget's end reports the last episode count
    samples = np.array([10, 20, 30])
    values = np.array([1.0, 1.5, 2.0])
    assert reaches_at(samples, values, 5.0, optimum=10.0, rnd=0.0) == 30
