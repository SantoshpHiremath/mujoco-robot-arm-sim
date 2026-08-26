import os

import mujoco
import pytest

from src.controller import PDGains
from src.kinematics import UnreachableTargetError
from src.reach import reach_target

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "two_link_arm.xml")


@pytest.fixture
def model():
    return mujoco.MjModel.from_xml_path(MODEL_PATH)


@pytest.fixture
def data(model):
    return mujoco.MjData(model)


class TestReachTarget:
    def test_reaches_target_within_small_error_under_real_dynamics(self, model, data):
        result = reach_target(model, data, target_x=0.5, target_z=0.6)
        assert result.settled
        # This tolerance reflects real simulated dynamics settling
        # under PD control -- not a restatement of the IK solution,
        # which is why it's checked against the simulator's own
        # end-effector site position (see reach.py's docstring).
        assert result.position_error < 0.02

    def test_unreachable_target_propagates_the_ik_error(self, model, data):
        with pytest.raises(UnreachableTargetError):
            reach_target(model, data, target_x=5.0, target_z=5.0)

    def test_elbow_up_and_elbow_down_both_reach_the_same_target(self, model, data):
        up = reach_target(model, data, target_x=0.4, target_z=0.7, elbow_up=True)
        mujoco.mj_resetData(model, data)
        down = reach_target(model, data, target_x=0.4, target_z=0.7, elbow_up=False)
        assert up.position_error < 0.02
        assert down.position_error < 0.02

    def test_stiffer_gains_still_converge_for_a_moderate_target(self, model, data):
        result = reach_target(model, data, target_x=0.3, target_z=0.55,
                               gains=PDGains(kp=80.0, kd=8.0))
        assert result.settled
        assert result.position_error < 0.02
