import math

import pytest

from src.kinematics import (
    LINK1_LENGTH, LINK2_LENGTH, BASE_HEIGHT,
    UnreachableTargetError, forward_kinematics, inverse_kinematics,
)


class TestForwardKinematics:
    def test_zero_angles_fully_extended_horizontal(self):
        p = forward_kinematics(0.0, 0.0)
        assert p.x == pytest.approx(LINK1_LENGTH + LINK2_LENGTH)
        assert p.z == pytest.approx(BASE_HEIGHT)

    def test_shoulder_ninety_degrees_elbow_straight(self):
        # Sign convention (verified against MuJoCo's own simulated
        # end-effector position in test_sim_matches_kinematics.py): a
        # positive shoulder angle rotates the arm *down* (-z), because
        # the model's hinge axis is "0 1 0" and MuJoCo's right-handed
        # frame rotates local +x toward -z for a positive rotation
        # about +y. This is not the "textbook" +z-up 2D convention.
        p = forward_kinematics(math.pi / 2, 0.0)
        assert p.x == pytest.approx(0.0, abs=1e-9)
        assert p.z == pytest.approx(BASE_HEIGHT - LINK1_LENGTH - LINK2_LENGTH)

    def test_folded_arm_elbow_180_degrees(self):
        # Elbow folded all the way back onto link 1 -> end effector
        # sits at distance (link1 - link2) from the base, along the
        # shoulder direction.
        p = forward_kinematics(0.0, math.pi)
        assert p.x == pytest.approx(LINK1_LENGTH - LINK2_LENGTH, abs=1e-9)
        assert p.z == pytest.approx(BASE_HEIGHT, abs=1e-9)


class TestInverseKinematics:
    def test_ik_solution_matches_forward_kinematics_elbow_up(self):
        target_x, target_z = 0.5, 0.6
        shoulder, elbow = inverse_kinematics(target_x, target_z, elbow_up=True)
        p = forward_kinematics(shoulder, elbow)
        assert p.x == pytest.approx(target_x, abs=1e-6)
        assert p.z == pytest.approx(target_z, abs=1e-6)

    def test_ik_solution_matches_forward_kinematics_elbow_down(self):
        target_x, target_z = 0.5, 0.6
        shoulder, elbow = inverse_kinematics(target_x, target_z, elbow_up=False)
        p = forward_kinematics(shoulder, elbow)
        assert p.x == pytest.approx(target_x, abs=1e-6)
        assert p.z == pytest.approx(target_z, abs=1e-6)

    def test_elbow_up_and_elbow_down_give_different_configurations(self):
        target_x, target_z = 0.5, 0.6
        shoulder_up, elbow_up = inverse_kinematics(target_x, target_z, elbow_up=True)
        shoulder_down, elbow_down = inverse_kinematics(target_x, target_z, elbow_up=False)
        assert elbow_up != pytest.approx(elbow_down)

    def test_fully_extended_target_is_reachable(self):
        max_reach = LINK1_LENGTH + LINK2_LENGTH
        shoulder, elbow = inverse_kinematics(max_reach, BASE_HEIGHT)
        p = forward_kinematics(shoulder, elbow)
        assert p.x == pytest.approx(max_reach, abs=1e-6)
        assert elbow == pytest.approx(0.0, abs=1e-6)

    def test_target_beyond_max_reach_raises(self):
        max_reach = LINK1_LENGTH + LINK2_LENGTH
        with pytest.raises(UnreachableTargetError):
            inverse_kinematics(max_reach + 0.5, BASE_HEIGHT)

    def test_target_inside_min_reach_raises(self):
        # min reach is |link1 - link2|; a target closer than that to
        # the base cannot be reached even with the elbow fully folded.
        with pytest.raises(UnreachableTargetError):
            inverse_kinematics(0.0, BASE_HEIGHT)

    def test_target_at_exactly_max_reach_boundary_does_not_raise(self):
        max_reach = LINK1_LENGTH + LINK2_LENGTH
        # Should not raise despite floating-point boundary proximity.
        inverse_kinematics(max_reach - 1e-9, BASE_HEIGHT)

    @pytest.mark.parametrize("target_x,target_z", [
        (0.3, 0.7), (-0.4, 0.5), (0.2, 0.3), (-0.2, 0.65), (0.6, 0.55),
    ])
    def test_ik_round_trip_across_multiple_targets(self, target_x, target_z):
        shoulder, elbow = inverse_kinematics(target_x, target_z, elbow_up=True)
        p = forward_kinematics(shoulder, elbow)
        assert p.x == pytest.approx(target_x, abs=1e-6)
        assert p.z == pytest.approx(target_z, abs=1e-6)
