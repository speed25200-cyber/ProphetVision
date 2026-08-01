"""Synthetic spin generator: renders physically simulated spins so the whole
pipeline can be validated end-to-end against known ground truth."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import WheelConfig


@dataclass
class SpinTruth:
    final_pocket: int
    final_index: int
    t_drop: float
    t_land: float
    impact_index: int
    scatter_offset: int


@dataclass
class SyntheticSpin:
    wheel: WheelConfig = field(default_factory=WheelConfig)
    size: int = 480
    fps: float = 60.0
    # True dynamics (unknown to the predictor).
    ball_omega0: float = 1900.0      # deg/s
    ball_c0: float = 170.0           # deg/s^2
    ball_c2: float = 6.0e-5          # 1/deg
    omega_drop_true: float = 640.0   # deg/s
    fall_time_true: float = 0.75
    fall_travel_true: float = 120.0  # deg
    rotor_omega: float = -110.0      # deg/s (opposite direction to ball)
    rotor_alpha: float = 1.5         # deg/s^2 toward zero speed
    scatter_sigma_pockets: float = 1.6
    seed: int = 0
    post_land_time: float = 0.6

    def simulate(self):
        """Returns (frames, times, truth). Frames are BGR uint8."""
        rng = np.random.default_rng(self.seed)
        n = self.wheel.n_pockets
        sector = 360.0 / n
        R = self.size * 0.45
        cx = cy = self.size / 2.0

        # Integrate ball rim phase.
        dt = 1.0 / (self.fps * 4)
        t, theta, omega = 0.0, rng.uniform(0, 360), self.ball_omega0
        phi = rng.uniform(0, 360)  # azimuth of zero pocket
        rotor_w = self.rotor_omega
        times, ball_states, rotor_states = [], [], []

        t_drop = None
        while omega > self.omega_drop_true:
            times.append(t)
            ball_states.append((theta, 0.92))
            rotor_states.append(phi)
            theta += omega * dt
            # deceleration with small measurement-free process noise
            omega -= (self.ball_c0 + self.ball_c2 * omega * omega) * dt
            phi += rotor_w * dt
            rotor_w += self.rotor_alpha * dt * (-np.sign(rotor_w))
            t += dt
        t_drop = t

        # Fall phase: ball spirals from rim to pocket ring.
        theta_impact = theta + np.sign(self.ball_omega0) * self.fall_travel_true
        n_fall = int(self.fall_time_true / dt)
        for i in range(n_fall):
            frac = (i + 1) / n_fall
            times.append(t)
            ball_states.append((theta + np.sign(self.ball_omega0)
                                * self.fall_travel_true * frac,
                                0.92 - 0.34 * frac))
            rotor_states.append(phi)
            phi += rotor_w * dt
            t += dt
        t_land = t

        # Impact pocket (relative to the zero mark, clockwise index order).
        impact_idx = int(round(((theta_impact - phi) % 360.0) / sector)) % n
        # Chaotic bounce: wrapped-Gaussian pocket offset in ball direction.
        offset = int(round(rng.normal(2.0, self.scatter_sigma_pockets)))
        final_idx = (impact_idx + int(np.sign(self.ball_omega0)) * offset) % n
        truth = SpinTruth(
            final_pocket=self.wheel.pocket_order[final_idx],
            final_index=final_idx, t_drop=t_drop, t_land=t_land,
            impact_index=impact_idx, scatter_offset=offset % n,
        )

        # Settled phase: ball co-rotates with the wheel in its pocket.
        n_post = int(self.post_land_time / dt)
        for _ in range(n_post):
            times.append(t)
            ball_states.append((phi + final_idx * sector, 0.55))
            rotor_states.append(phi)
            phi += rotor_w * dt
            t += dt

        # Render at fps (subsample the dt grid).
        stride = 4
        frames, frame_times = [], []
        for i in range(0, len(times), stride):
            frames.append(self._render(ball_states[i], rotor_states[i],
                                       cx, cy, R, sector))
            frame_times.append(times[i])
        return frames, np.asarray(frame_times), truth

    # ------------------------------------------------------------------
    def _render(self, ball_state, phi, cx, cy, R, sector) -> np.ndarray:
        img = np.full((self.size, self.size, 3), 30, np.uint8)
        cv2.circle(img, (int(cx), int(cy)), int(R * 1.04), (60, 60, 60), -1)
        cv2.circle(img, (int(cx), int(cy)), int(R * 0.70), (40, 40, 45), -1)
        # Pocket ring sectors between 0.45R and 0.65R.
        for i in range(self.wheel.n_pockets):
            a0 = phi + i * sector - sector / 2
            if i == 0:
                color = (40, 160, 40)
            elif i % 2 == 1:
                color = (40, 40, 170)
            else:
                color = (25, 25, 25)
            cv2.ellipse(img, (int(cx), int(cy)),
                        (int(R * 0.65), int(R * 0.65)), 0, a0, a0 + sector,
                        color, -1)
        cv2.circle(img, (int(cx), int(cy)), int(R * 0.45), (70, 70, 75), -1)
        # Ball.
        theta_deg, r_frac = ball_state
        bx = cx + R * r_frac * np.cos(np.radians(theta_deg))
        by = cy + R * r_frac * np.sin(np.radians(theta_deg))
        cv2.circle(img, (int(bx), int(by)), max(3, self.size // 120),
                   (250, 250, 250), -1)
        return img


def write_video(path: str, frames: list[np.ndarray], fps: float) -> None:
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    if not vw.isOpened():
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
