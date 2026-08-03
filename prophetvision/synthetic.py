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
    # --- oblique ("side view") rendering ------------------------------
    # 0 keeps the top-down render. Otherwise the wheel plane is projected as
    # seen from `elevation_deg` above it, and the frame is dimmed and noised
    # to reproduce the low-SNR regime of a wide side-view shot, where
    # per-frame blob detection of the ball fails.
    elevation_deg: float = 0.0
    side_gain: float = 0.45
    side_noise: float = 6.0

    def _oblique_H(self) -> np.ndarray:
        """Homography of the wheel plane seen from `elevation_deg`."""
        e = np.radians(max(self.elevation_deg, 1e-3))
        s = self.size
        c = s / 2.0
        # Scaled-orthographic tilt about the image x axis, plus a mild
        # perspective term so the near rim is larger than the far rim.
        k = 0.35 * np.cos(e)
        H = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.sin(e), 0.0],
            [0.0, k / s, 1.0],
        ], dtype=np.float64)
        # keep the wheel centred
        T1 = np.array([[1, 0, -c], [0, 1, -c], [0, 0, 1]], dtype=np.float64)
        T2 = np.array([[1, 0, c], [0, 1, c], [0, 0, 1]], dtype=np.float64)
        return T2 @ H @ T1

    def _to_oblique(self, img: np.ndarray, rng) -> np.ndarray:
        warped = cv2.warpPerspective(img, self._oblique_H(),
                                     (self.size, self.size),
                                     flags=cv2.INTER_LINEAR)
        out = warped.astype(np.float32) * self.side_gain
        out += rng.normal(0.0, self.side_noise, out.shape)
        return np.clip(out, 0, 255).astype(np.uint8)

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
            img = self._render(ball_states[i], rotor_states[i],
                               cx, cy, R, sector)
            if self.elevation_deg > 0:
                img = self._to_oblique(img, rng)
            frames.append(img)
            frame_times.append(times[i])
        return frames, np.asarray(frame_times), truth

    def oblique_mapping(self):
        """(r_frac, theta_deg) -> image pixel, matching `elevation_deg`."""
        H = self._oblique_H()
        c = self.size / 2.0
        R = self.size * 0.45

        def to_image(r_frac, theta_deg):
            th = np.radians(np.asarray(theta_deg, dtype=float))
            x = c + R * np.asarray(r_frac) * np.cos(th)
            y = c + R * np.asarray(r_frac) * np.sin(th)
            den = H[2, 0] * x + H[2, 1] * y + H[2, 2]
            return ((H[0, 0] * x + H[0, 1] * y + H[0, 2]) / den,
                    (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / den)
        return to_image

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
        # Cream-yellow like a real roulette ball (exercises color trackers).
        cv2.circle(img, (int(bx), int(by)), max(3, self.size // 120),
                   (150, 230, 235), -1)
        return img


def write_video(path: str, frames: list[np.ndarray], fps: float) -> None:
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    if not vw.isOpened():
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
