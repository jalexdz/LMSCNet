import numpy as np

class RandomTransform:
    def __init__(self, rotation_range=np.pi, translation_range=0.1):
        self.rotation_range = rotation_range
        self.translation_range = translation_range

    def _get_transform(self):
        # Random rotation about Z (assuming upright scenes)
        theta = np.random.uniform(-self.rotation_range, self.rotation_range)
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        R = np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta,  cos_theta, 0],
            [0,               0,    1]
        ])

        # Random translation
        t = np.random.uniform(-self.translation_range, self.translation_range, size=(3,))

        return R, t
    
    def __call__(self, pts, labels=None):
        """
        pts: (N, 3) numpy array
        labels: optional (N,) array

        returns transformed pts and labels (if provided)
        """
        R, t = self._get_transform()
        pts_transformed = pts @ R.T + t

        if labels is not None:
            return pts_transformed, labels
        else:
            return pts_transformed