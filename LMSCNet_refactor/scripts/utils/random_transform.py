import numpy as np

class RandomTransform:
    def __init__(self, rotation_range=np.pi, translation_range=0.1):
        self.rotation_range = rotation_range
        self.translation_range = translation_range

    def sample(self):
        theta = np.random.uniform(-self.rotation_range, self.rotation_range)
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        R = np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta,  cos_theta, 0],
            [0,               0,    1]
        ])
        t = np.random.uniform(-self.translation_range, self.translation_range, size=(3,))
        return R, t

    def apply(self, pts, R, t):
        return pts @ R.T + t

    def __call__(self, pts, labels=None, R=None, t=None):
        if R is None or t is None:
            R, t = self.sample()
        pts_transformed = self.apply(pts, R, t)
        return (pts_transformed, labels) if labels is not None else pts_transformed
