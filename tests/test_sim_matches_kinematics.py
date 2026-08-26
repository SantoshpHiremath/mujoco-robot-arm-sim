"""
The single most important verification in this project: confirms the
hand-written analytical forward_kinematics() function actually agrees
with MuJoCo's own physics engine's reported end-effector position, for
a real loaded MJCF model. Without this test, the kinematics module
would just be "math that looks right" -- this is what makes it a
checked, trustworthy cross-reference against the actual simulator.
"""
import math
import os

import mujoco
import pytest

from src.kinematics import forward_kinematics

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "two_link_arm.xml")


@pytest.fixture
def model():
    return mujoco.MjModel.from_xml_path(MODEL_PATH)


@pytest.fixture
def data(model):
    return mujoco.MjData(model)


def _sim_ee_position(model, data, shoulder_angle, elbow_angle):
    data.qpos[0] = shoulder_angle
    data.qpos[1] = elbow_angle
    mujoco.mj_forward(model, data)
    ee = data.site("ee_site").xpos
    return float(ee[0]), float(ee[2])


class TestSimulatorMatchesAnalyticalKinematics:
    @pytest.mark.parametrize("shoulder,elbow", [
        (0.0, 0.0),
        (math.pi / 2, 0.0),
        (0.0, math.pi / 2),
        (0.7, -0.5),
        (-1.2, 1.8),
        (math.pi, -math.pi / 3),
    ])
    def test_analytical_fk_matches_mujoco_site_position(self, model, data, shoulder, elbow):
        sim_x, sim_z = _sim_ee_position(model, data, shoulder, elbow)
        analytical = forward_kinematics(shoulder, elbow)
        assert analytical.x == pytest.approx(sim_x, abs=1e-6)
        assert analytical.z == pytest.approx(sim_z, abs=1e-6)

    def test_model_has_expected_joint_count_and_names(self, model):
        assert model.njnt == 2
        assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, 0) == "shoulder"
        assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, 1) == "elbow"
