import numpy as np

class JointPID:
    """Per-joint PID with velocity feedforward, anti-windup and torque clipping."""
    def __init__(self, kp, ki, kd, dt, tau_max, i_max=None):
        self.kp, self.ki, self.kd = map(np.asarray, (kp, ki, kd))
        self.dt = dt
        self.tau_max = np.asarray(tau_max)
        self.i_max = np.asarray(i_max) if i_max is not None else 0.5 * self.tau_max / np.maximum(self.ki, 1e-9)
        self.reset()

    def reset(self):
        self.integral = np.zeros_like(self.kp, dtype=float)

    def __call__(self, q_ref, q, qd, qd_ref=None):
        if qd_ref is None:
            qd_ref = np.zeros_like(q)
        e = q_ref - q
        e_dot = qd_ref - qd
        self.integral = np.clip(self.integral + e * self.dt, -self.i_max, self.i_max)  # anti-windup
        tau = self.kp * e + self.ki * self.integral + self.kd * e_dot
        return np.clip(tau, -self.tau_max, self.tau_max)