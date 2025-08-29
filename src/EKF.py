import numpy as np

def wrap_angle(a):
    """Wrap angle to (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi

class EKF:
    """
    State:
        x = [px, py, theta, v, b_a, b_d]

        px [m]      : x pose in the global frame
        py [m]      : y pose in the global frame
        theta [rad] : yaw in global frame
        v [m/s]     : velocity
        b_a [m/s^2] : Accelerometer bias
        b_w [rad/s] : Gyroscope bias

    Input:
        u_meas = [a_meas, w_meas]
    Slam measurment:
        z_slam = [px, py, th]
    Tone wheels measurment:
        z_tone = [v]
    """

    def __init__(self, L):
        self.L = L
        self.n = 6
        self.m_gps = 2
        self.m_slam = 3
        self.x = np.zeros(self.n)
        self.P = np.eye(self.n)
        # Initial tuning 
        self.Q = np.diag([1e-3, 1e-3, 2e-2, 2e-1, 1e-5, 1e-5])  # processo [px, py, theta, v, ba, bw]
        self.R_tone = np.diag([0.01]) # Tone (sigma ~0.5m/s)
        self.R_slam = np.diag([0.22, 0.22, 0.1]) # SLAM ( 0.1-10m , 0.001-0.01rad)

        # NIS (Normalized Innovation Squared)
        # self.nis_slam = []
        # self.nis_tone = []

    def set_state(self, x0, P0=None):
        self.x = x0.copy()
        if P0 is not None:
            self.P = P0.copy()

    def predict(self, u_meas, dt):
        px, py, th, v, b_a, b_w = self.x
        a_m, w_m = u_meas

        # bias compensation
        a = a_m - b_a
        w = w_m - b_w

        # discrete dynamics using kinematic bicycle model
        px_n = px + v * np.cos(th) * dt
        py_n = py + v * np.sin(th) * dt
        th_n = wrap_angle(th + w * dt)
        v_n = v + a * dt
        b_a_n = b_a
        b_w_n = b_w

        x_pred = np.array([px_n, py_n, th_n, v_n, b_a_n, b_w_n])

        # Jacobian F = df/dx
        c, s = np.cos(th), np.sin(th)
        F = np.eye(self.n)
        # px wrt th and v
        F[0, 2] = -v * s * dt
        F[0, 3] = c * dt
        # py wrt th and v
        F[1, 2] = v * c * dt
        F[1, 3] = s * dt
        # theta wrt b_w
        F[2, 5] = -dt
        # v wrt b_a
        F[3, 4] = -dt
        # (bias states are random walks → diag already = 1)

        # Covariance propagation
        P_pred = F @ self.P @ F.T + self.Q

        # Commit
        self.x = x_pred
        self.P = P_pred

    def update_slam(self, z):
        # h(x) = [px, py]
        H = np.zeros((self.m_slam, self.n))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0

        z_pred = np.array([self.x[0], self.x[1], self.x[2]])
        y = z - z_pred
        y[2] = wrap_angle(y[2])  # handle wrap around
        S = H @ self.P @ H.T + self.R_slam
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(self.n)
        self.P = (I - K @ H) @ self.P

        # normalize angle
        self.x[2] = wrap_angle(self.x[2])

        # nis
        # v = z - z_pred
        # self.nis_slam.append(v @ np.linalg.inv(S) @ v.T)

    def update_tone_wheels(self, z):
        # h(x) = [vx]
        H = np.zeros((1, self.n))
        H[0, 3] = 1.0

        z_pred = np.array([self.x[3]])

        y = z - z_pred
        S = H @ self.P @ H.T + self.R_tone
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(self.n)
        self.P = (I - K @ H) @ self.P

        # normalize angolo
        self.x[2] = wrap_angle(self.x[2])

        # nis
        # v = z - z_pred
        # self.nis_tone.append(v @ np.linalg.inv(S) @ v.T)
