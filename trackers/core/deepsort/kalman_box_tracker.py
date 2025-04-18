from typing import Optional, Union

import numpy as np

from trackers.core.sort.kalman_box_tracker import SORTKalmanBoxTracker


class DeepSORTKalmanBoxTracker(SORTKalmanBoxTracker):
    """
    The `DeepSORTKalmanBoxTracker` class represents the internals of a single
    tracked object (bounding box), with a Kalman filter to predict and update
    its position. It also maintains a feature vector for the object, which is
    used to identify the object across frames.

    Attributes:
        tracker_id (int): Unique identifier for the tracker.
        number_of_successful_updates (int): Number of times the object has been
            updated successfully.
        time_since_update (int): Number of frames since the last update.
        state (np.ndarray): State vector of the bounding box.
        F (np.ndarray): State transition matrix.
        H (np.ndarray): Measurement matrix.
        Q (np.ndarray): Process noise covariance matrix.
        R (np.ndarray): Measurement noise covariance matrix.
        P (np.ndarray): Error covariance matrix.
        features (list[np.ndarray]): List of feature vectors.
        count_id (int): Class variable to assign unique IDs to each tracker.

    Args:
        bbox (np.ndarray): Initial bounding box in the form [x1, y1, x2, y2].
        feature (Optional[np.ndarray]): Optional initial feature vector.
    """

    count_id = 0

    @classmethod
    def get_next_tracker_id(cls) -> int:
        """
        Class method that returns the next available tracker ID.

        Returns:
            int: The next available tracker ID.
        """
        next_id = cls.count_id
        cls.count_id += 1
        return next_id

    def __init__(self, bbox: np.ndarray, feature: Optional[np.ndarray] = None):
        # Call the parent class constructor to handle the basic tracker functionality
        super().__init__(bbox)

        # Initialize features list
        self.features: list[np.ndarray] = []
        if feature is not None:
            self.features.append(feature)

    def update_feature(self, feature: np.ndarray):
        self.features.append(feature)

    def get_feature(self) -> Union[np.ndarray, None]:
        """
        Get the mean feature vector for this tracker.

        Returns:
            np.ndarray: Mean feature vector.
        """
        if len(self.features) > 0:
            # Return the mean of all features, thus (in theory) capturing the
            # "average appearance" of the object, which should be more robust
            # to minor appearance changes. Otherwise, the last feature can
            # also be returned like the following:
            # return self.features[-1]
            return np.mean(self.features, axis=0)
        return None
