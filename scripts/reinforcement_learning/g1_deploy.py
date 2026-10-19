# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a LEAPP-exported policy on a physical Unitree G1.

Inference is the official ``leapp.InferenceManager``. This script supplies the two ends the graph
cannot reach by itself: the robot's state and camera on the way in, its motors on the way out. No
simulator is launched and Isaac Sim is never imported.

Needs ``unitree_sdk2py`` with ``cyclonedds==0.10.2``, and a depth publisher on the robot's onboard
computer sending 64x38 frames as unsigned 16-bit millimetres over UDP.
"""

from __future__ import annotations

import argparse
import math
import re
import select
import socket
import sys
import threading
import time

import numpy as np

CONTROL_DT = 0.02
"""Control period [s]. The exported graph carries its history at this rate."""

DEPTH_SHAPE = (38, 64)
"""Depth frames the publisher sends and the graph expects, as ``(height, width)``."""

DEPTH_MAX_RANGE = 3.0
"""Range the graph clips at [m], and what a pixel with no return is filled with."""

DEPTH_MAX_AGE = 0.2
"""Refuse to step on a frame older than this [s]. Ten control periods."""

MOTOR_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
"""The 29 body joints ``rt/lowcmd`` carries, in its own order.

The exported graph writes every joint of the articulation, fingers included, so its outputs are
matched to these by name rather than by position -- the two orders agree for the first 22 joints
and then diverge, which is the kind of mismatch that produces a walking robot with the wrong arm.
"""

DEFAULT_JOINT_POS = {
    ".*_hip_pitch_joint": -0.20,
    ".*_knee_joint": 0.42,
    ".*_ankle_pitch_joint": -0.23,
    ".*_elbow_joint": 0.87,
    "left_shoulder_pitch_joint": 0.35,
    "left_shoulder_roll_joint": 0.16,
    "right_shoulder_pitch_joint": 0.35,
    "right_shoulder_roll_joint": -0.16,
}
"""The pose the ramp ends at [rad], by joint-name pattern; joints not listed go to zero.

This is the task's own initial state, and it is fixed on purpose. The operator squares the robot up
in this pose and it has to be the same every run -- a target derived from the policy's output would
depend on whatever pose the robot was sagging in when it was read.
"""

RAMP_KP = 100.0
"""Stiffness to reach and hold the start pose with [N*m/rad], until the policy supplies its own."""

RAMP_KD = 2.0
"""Damping for the same [N*m*s/rad]."""


class DepthStream:
    """Newest depth frame from the publisher, in metres.

    The publisher sends whole frames in one datagram, so a frame either arrives intact or not at
    all. What this guards is staleness, not corruption: a dead publisher leaves the last frame in
    place and the array alone cannot say how old it is.
    """

    def __init__(self, port: int) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("", port))
        self._sock.settimeout(0.2)
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._at = 0.0
        threading.Thread(target=self._receive, daemon=True).start()

    def _receive(self) -> None:
        count = DEPTH_SHAPE[0] * DEPTH_SHAPE[1]
        while True:
            try:
                payload = self._sock.recv(1 << 16)
            except (TimeoutError, socket.timeout):
                continue
            if len(payload) < count * 2:
                continue
            millimetres = np.frombuffer(payload, dtype=np.uint16, count=count, offset=len(payload) - count * 2)
            frame = millimetres.reshape(DEPTH_SHAPE).astype(np.float32) / 1000.0
            # A real camera reports no return as zero. The graph reads zero as a surface against
            # the lens, so the clip range is what says "nothing within range" instead.
            frame[frame <= 0.0] = DEPTH_MAX_RANGE
            with self._lock:
                self._frame, self._at = np.clip(frame, 0.0, DEPTH_MAX_RANGE), time.monotonic()

    def wait(self, timeout: float = 20.0) -> None:
        """Block until the first frame arrives, so a missing camera stops the run before the ramp."""
        deadline = time.monotonic() + timeout
        while self.latest() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        if self.latest() is None:
            raise TimeoutError(f"no depth frame in {timeout:.0f}s; is the publisher running?")

    def latest(self) -> np.ndarray | None:
        """Newest frame [m], or ``None`` when it is missing or stale."""
        with self._lock:
            if self._frame is None or time.monotonic() - self._at > DEPTH_MAX_AGE:
                return None
            return self._frame


class Teleop:
    """Forward speed and heading from single keypresses, without waiting for Enter."""

    HELP = "w/s forward +/-   q/e turn left/right   space stop   x quit"
    SPEED_STEP = 0.1
    HEADING_STEP = math.radians(15.0)

    def __init__(self) -> None:
        self.vx = 0.0
        self.heading = 0.0
        self.running = True

    def _poll(self) -> str | None:
        if not select.select([sys.stdin], [], [], 0.0)[0]:
            return None
        return sys.stdin.read(1)

    def command(self, quat_xyzw: np.ndarray) -> np.ndarray:
        """``(vx, vy, yaw_rate)`` [m/s, m/s, rad/s], with the yaw rate steering toward a heading.

        The command's third element is a rate, not a heading, so the heading target is turned into
        one here against the measured orientation -- the same controller the task trains with.
        """
        key = self._poll()
        if key == "w":
            self.vx = min(1.0, self.vx + self.SPEED_STEP)
        elif key == "s":
            self.vx = max(0.0, self.vx - self.SPEED_STEP)
        elif key == "q":
            self.heading += self.HEADING_STEP
        elif key == "e":
            self.heading -= self.HEADING_STEP
        elif key == " ":
            self.vx = 0.0
        elif key == "x":
            self.running = False
        if key:
            print(f"[cmd] vx={self.vx:+.2f} m/s  heading={math.degrees(self.heading):+.0f} deg")

        x, y, z, w = quat_xyzw
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        error = (self.heading - yaw + math.pi) % (2.0 * math.pi) - math.pi
        return np.array([self.vx, 0.0, np.clip(error, -1.0, 1.0)], dtype=np.float32)


class G1Link:
    """``rt/lowstate`` in, ``rt/lowcmd`` out, over the Unitree SDK."""

    def __init__(self, domain_id: int, interface: str) -> None:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC

        ChannelFactoryInitialize(domain_id, interface)
        self._state: LowState_ | None = None
        self._subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self._subscriber.Init(self._on_state, 10)
        self._publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self._publisher.Init()
        self._command = LowCmd_()
        self._crc = CRC()
        self._next = time.monotonic()
        while self._state is None:
            time.sleep(0.02)

    def _on_state(self, message) -> None:
        self._state = message

    def read(self):
        """Joint state and IMU, already in the frames the graph expects."""
        state = self._state
        motors = state.motor_state
        quat_wxyz = np.asarray(state.imu_state.quaternion[:4], dtype=np.float32)
        return type(
            "G1State",
            (),
            {
                "joint_pos": np.array([m.q for m in motors[: len(MOTOR_NAMES)]], dtype=np.float32),
                "joint_vel": np.array([m.dq for m in motors[: len(MOTOR_NAMES)]], dtype=np.float32),
                # The IMU publishes (w, x, y, z); the graph expects (x, y, z, w).
                "root_quat_xyzw": np.roll(quat_wxyz, -1),
                "ang_vel": np.asarray(state.imu_state.gyroscope[:3], dtype=np.float32),
                "mode_machine": int(state.mode_machine),
            },
        )

    def write(self, joint_pos, kp, kd) -> None:
        """Send one ``rt/lowcmd``. The gains are the policy's own outputs, not configuration."""
        for i in range(len(MOTOR_NAMES)):
            motor = self._command.motor_cmd[i]
            motor.mode, motor.q, motor.dq, motor.tau = 1, float(joint_pos[i]), 0.0, 0.0
            motor.kp, motor.kd = float(kp[i]), float(kd[i])
        self._command.mode_machine = self._state.mode_machine
        self._command.crc = self._crc.Crc(self._command)
        self._publisher.Write(self._command)

    def damp(self, kd, seconds: float = 1.5) -> None:
        """Bleed off energy with no stiffness, then stop commanding. Every exit goes through here."""
        print("[..] damping down")
        zero = np.zeros(len(MOTOR_NAMES), dtype=np.float32)
        for _ in range(int(seconds / CONTROL_DT)):
            self.write(zero, zero, kd)
            self.wait(CONTROL_DT)

    def wait(self, period: float) -> None:
        """Sleep until the next control instant, without accumulating drift."""
        self._next += period
        time.sleep(max(0.0, self._next - time.monotonic()))


def default_pose() -> np.ndarray:
    """:data:`DEFAULT_JOINT_POS` resolved onto the motors, in ``rt/lowcmd`` order [rad]."""
    pose = np.zeros(len(MOTOR_NAMES), dtype=np.float32)
    for pattern, value in DEFAULT_JOINT_POS.items():
        for i, name in enumerate(MOTOR_NAMES):
            if re.fullmatch(pattern, name):
                pose[i] = value
    return pose


def graph_to_motors(pipeline: str) -> np.ndarray:
    """Index into the graph's ``joint_pos`` output for each motor, read from the export.

    Args:
        pipeline: The exported LEAPP ``.yaml``.

    Returns:
        ``(29,)`` indices, so ``outputs["joint_pos"][index]`` is in ``rt/lowcmd`` order.

    Raises:
        KeyError: If the export does not drive one of the motors, rather than silently leaving it
            at whatever the array happened to hold.
    """
    import yaml

    description = yaml.safe_load(open(pipeline))
    model = next(iter(description["models"].values()))
    names = next(o for o in model["outputs"] if o["name"] == "joint_pos")["element_names"][0]
    order = {name: i for i, name in enumerate(names)}
    missing = [name for name in MOTOR_NAMES if name not in order]
    if missing:
        raise KeyError(f"the export does not drive {missing}")
    return np.array([order[name] for name in MOTOR_NAMES], dtype=np.int64)


def confirm(word: str, prompt: str) -> None:
    """Block until the operator types ``word``. Anything else aborts."""
    print(prompt)
    if input(f">>> type '{word}' and press Enter: ").strip() != word:
        raise SystemExit("[..] aborted by the operator")


def bring_up(link: G1Link, default_pose: np.ndarray, kp, kd, ramp_s: float) -> None:
    """Interpolate to the policy's start pose and hold it until the operator engages the policy.

    Two steps rather than one, and ``rt/lowcmd`` keeps flowing through both: the robot is lowered
    onto its feet between them, and a gap in the command stream there would drop it.
    """
    confirm("go", "\n>>> Robot supported, area clear, e-stop in hand.")
    start = link.read().joint_pos.copy()
    steps = int(ramp_s / CONTROL_DT)
    print(f"[..] ramping to the start pose over {ramp_s:.1f}s")
    for step in range(steps):
        link.write(start + (default_pose - start) * (step + 1) / steps, kp, kd)
        link.wait(CONTROL_DT)

    print("\n>>> HOLDING the start pose. Square the robot up and lower it onto its feet.")
    print(">>> KEEP IT SUPPORTED: this is PD with no policy, and the pose is not statically stable.")
    held = threading.Event()
    threading.Thread(target=lambda: (input(">>> type 'go2' and press Enter: "), held.set()), daemon=True).start()
    while not held.is_set():
        link.write(default_pose, kp, kd)
        link.wait(CONTROL_DT)
    print("[ok] engaging in 3s -- hands clear")
    for _ in range(int(3.0 / CONTROL_DT)):
        link.write(default_pose, kp, kd)
        link.wait(CONTROL_DT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pipeline", required=True, help="Exported LEAPP .yaml.")
    parser.add_argument("--interface", required=True, help="Network interface on the robot's subnet.")
    parser.add_argument("--domain_id", type=int, default=0, help="DDS domain.")
    parser.add_argument("--depth_port", type=int, default=5601, help="UDP port the publisher sends to.")
    args = parser.parse_args()

    from leapp import InferenceManager

    inference = InferenceManager(args.pipeline)
    link = G1Link(args.domain_id, args.interface)
    depth = DepthStream(args.depth_port)
    teleop = Teleop()
    depth.wait()

    motors = graph_to_motors(args.pipeline)
    ramp_kp = np.full(len(MOTOR_NAMES), RAMP_KP, dtype=np.float32)
    ramp_kd = np.full(len(MOTOR_NAMES), RAMP_KD, dtype=np.float32)
    print(f"[ok] mode_machine={link.read().mode_machine}  policy at {1 / CONTROL_DT:.0f} Hz")
    print(f"[..] teleop: {Teleop.HELP}")

    bring_up(link, default_pose(), ramp_kp, ramp_kd, ramp_s=3.0)
    try:
        while teleop.running:
            state = link.read()
            frame = depth.latest()
            if frame is None:
                raise RuntimeError(f"no depth frame within {DEPTH_MAX_AGE * 1e3:.0f} ms")
            outputs = inference.run_policy({
                "robot_joint_pos": state.joint_pos,
                "robot_joint_vel": state.joint_vel,
                "robot_root_quat_w": state.root_quat_xyzw,
                "robot_root_ang_vel_b": state.ang_vel,
                "base_velocity": teleop.command(state.root_quat_xyzw),
                "depth_camera_output_distance_to_image_plane": frame,
            })
            link.write(
                joint_pos=np.asarray(outputs["joint_pos"]).reshape(-1)[motors],
                kp=np.asarray(outputs["joint_pos_kp_gains"]).reshape(-1)[motors],
                kd=np.asarray(outputs["joint_pos_kd_gains"]).reshape(-1)[motors],
            )
            link.wait(CONTROL_DT)
    except (KeyboardInterrupt, RuntimeError) as error:
        print(f"\n[..] {error or 'interrupted'}")
    finally:
        link.damp(ramp_kd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
