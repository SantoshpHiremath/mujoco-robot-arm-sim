import os

import mujoco
import numpy as np
import pytest

from src.controller import PDGains, pd_torque, run_to_target

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "two_link_arm.xml")


@pytest.fixture
def model():
    return mujoco.MjModel.from_xml_path(MODEL_PATH)


@pytest.fixture
def data(model):
    return mujoco.MjData(model)


class TestPDTorque:
    def test_zero_error_zero_velocity_gives_zero_torque(self):
        assert pd_torque(target_angle=1.0, current_angle=1.0, current_velocity=0.0,
                          gains=PDGains(kp=10, kd=1)) == 0.0

    def test_positive_error_gives_positive_torque(self):
        torque = pd_torque(target_angle=1.0, current_angle=0.0, current_velocity=0.0,
                            gains=PDGains(kp=10, kd=1))
        assert torque > 0

    def test_velocity_damps_torque(self):
        # Moving toward the target should reduce commanded torque
        # relative to being stationary at the same position error.
        stationary = pd_torque(1.0, 0.0, 0.0, PDGains(kp=10, kd=1))
        moving_toward_target = pd_torque(1.0, 0.0, 0.5, PDGains(kp=10, kd=1))
        assert moving_toward_target < stationary


class TestRunToTarget:
    def test_arm_settles_at_a_reachable_target_from_rest(self, model, data):
        target = np.array([0.5, -0.3])
        result = run_to_target(model, data, target_qpos=target,
                                gains=PDGains(kp=40, kd=4), max_steps=5000)
        assert result.settled, "Arm did not settle within max_steps"
        assert np.allclose(result.final_qpos[:2], target, atol=0.02)

    def test_settling_step_count_is_positive_and_bounded(self, model, data):
        target = np.array([0.3, 0.2])
        result = run_to_target(model, data, target_qpos=target,
                                gains=PDGains(kp=40, kd=4), max_steps=5000)
        assert result.settled
        assert 0 < result.settled_at_step <= 5000

    def test_higher_kd_reduces_or_maintains_overshoot(self, model, data):
        # A real, checkable control-theory property: more derivative
        # damping should not make a step response wilder. Measured via
        # the recorded qpos history's max overshoot past the target.
        target = np.array([1.0, 0.0])

        mujoco.mj_resetData(model, data)
        low_damping = run_to_target(model, data, target_qpos=target,
                                     gains=PDGains(kp=60, kd=1), max_steps=3000,
                                     record_history=True)
        mujoco.mj_resetData(model, data)
        high_damping = run_to_target(model, data, target_qpos=target,
                                      gains=PDGains(kp=60, kd=15), max_steps=3000,
                                      record_history=True)

        def max_overshoot(hist, target_val):
            shoulder_positions = [q[0] for q in hist]
            over = [p - target_val for p in shoulder_positions if p > target_val]
            return max(over) if over else 0.0

        low_overshoot = max_overshoot(low_damping.qpos_history, target[0])
        high_overshoot = max_overshoot(high_damping.qpos_history, target[0])
        assert high_overshoot <= low_overshoot + 1e-6

    def test_does_not_settle_if_max_steps_too_small(self, model, data):
        target = np.array([1.0, -1.0])
        result = run_to_target(model, data, target_qpos=target,
                                gains=PDGains(kp=40, kd=4), max_steps=3)
        assert not result.settled
        assert result.settled_at_step is None
        assert result.n_steps_run == 3
