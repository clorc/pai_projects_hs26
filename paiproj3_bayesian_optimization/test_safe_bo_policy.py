"""Focused policy checks: python -m unittest test_safe_bo_policy -v."""

import unittest

import numpy as np

from solution import SafeBOAgent


class TableGP:
    """Controlled posterior predictions at the test grid's integer inputs."""

    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=float)
        self.std = np.asarray(std, dtype=float)

    def predict(self, candidates, return_std=False):
        indices = np.rint(candidates[:, 0]).astype(int)
        mean = self.mean[indices].copy()
        return (mean, self.std[indices].copy()) if return_std else mean


def make_agent(probability):
    agent = SafeBOAgent()
    agent.exploration_probability = probability
    agent.candidate_grid = np.arange(5.0)
    agent.observe([[1.0]], 0.9, 3.0)
    agent.gpr_estim_f = TableGP(
        mean=[0.0, 0.9, 0.0, 0.0, 0.0],
        std=[0.05, 0.01, 0.2, 100.0, 10.0],
    )
    # x=3 is unsafe and separates the seed's safe region from x=4.
    # x=0 has substantial constraint uncertainty, but its UCB is still safe.
    agent.gpr_estim_v = TableGP(
        mean=np.array([2.0, 3.0, 3.0, 5.0, 3.0]) - 4.0,
        std=[0.4, 0.01, 0.01, 0.01, 0.01],
    )
    return agent


class SafeBOPolicyTests(unittest.TestCase):
    def test_zero_probability_preserves_ucb_choice(self):
        agent = make_agent(0.0)
        np.testing.assert_array_equal(agent.optimize_acquisition(), [[1.0]])
        self.assertEqual(agent.last_selection_mode, "ucb")

    def test_exploration_uses_both_outputs_and_respects_safe_region(self):
        agent = make_agent(1.0)
        # Neither the unsafe x=3 nor the disconnected x=4 is admissible,
        # despite their much higher objective uncertainty.
        np.testing.assert_array_equal(agent.optimize_acquisition(), [[0.0]])
        self.assertEqual(agent.last_selection_mode, "uncertainty")
        # Remove x=0's constraint uncertainty; the exploration choice changes.
        agent.gpr_estim_v.std[0] = 0.0
        np.testing.assert_array_equal(agent.optimize_acquisition(), [[2.0]])

    def test_trusted_safe_seed_fallback_in_both_modes(self):
        for probability in (0.0, 1.0):
            with self.subTest(probability=probability):
                agent = make_agent(probability)
                agent.gpr_estim_v.mean[:] = 1.0  # SA = 5 after restoring the mean.
                np.testing.assert_array_equal(agent.optimize_acquisition(), [[1.0]])

    def test_local_seed_reproduces_mixed_policy(self):
        first, second = make_agent(0.2), make_agent(0.2)
        traces = []
        for agent in (first, second):
            trace = []
            for _ in range(30):
                point = agent.optimize_acquisition()
                trace.append((agent.last_selection_mode, point.item()))
            traces.append(trace)
        self.assertEqual(traces[0], traces[1])
        self.assertEqual({mode for mode, _ in traces[0]}, {"ucb", "uncertainty"})

    def test_invalid_probability_is_rejected(self):
        for probability in (-0.1, 1.1):
            with self.subTest(probability=probability):
                with self.assertRaises(ValueError):
                    make_agent(probability).optimize_acquisition()

    def test_early_checker_boundary_violation_is_rejected(self):
        # A recorded checker point had SA=4.0272, although its previous
        # three-standard-deviation upper bound was 3.9938. The stronger
        # safety rule must reject it before an observation is received.
        agent = SafeBOAgent()
        agent.observe([[2.9232308077019256]], 0.22441265086699377, 3.496655884669586)
        agent._fit_models()
        self.assertFalse(np.isfinite(agent.acquisition_function(np.array([[2.87]]))).item())


if __name__ == "__main__":
    unittest.main()
