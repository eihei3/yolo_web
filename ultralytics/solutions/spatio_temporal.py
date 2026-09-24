# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import spatial

from ultralytics.solutions.solutions import BaseSolution, SolutionResults


class SpatioTemporalAnalyzer(BaseSolution):
    """
    A class for spatio-temporal analysis of object tracks.

    This class extends BaseSolution to provide functionality for analyzing
    spatial and temporal patterns in object tracking data, including:
    - Trajectory analysis
    - Speed estimation
    - Path prediction
    - Object interaction detection
    - Congestion analysis

    Attributes:
        history_length (int): Number of previous positions to store for each track.
        speed_threshold (float): Threshold for detecting high-speed objects.
        interaction_distance (float): Distance threshold for detecting object interactions.
        prediction_horizon (int): Number of frames to predict ahead.
        track_history (Dict[int, List[Tuple[float, float, float]]]): History of track positions and timestamps.
        speed_data (Dict[int, float]): Current speed of each track.
        congestion_areas (List[Dict]): Areas of congestion detected in the scene.

    Methods:
        process: Process image data and perform spatio-temporal analysis.
        update_track_history: Update the history of track positions.
        estimate_speed: Estimate the speed of each track.
        predict_path: Predict the future path of each track.
        detect_interactions: Detect interactions between objects.
        analyze_congestion: Analyze congestion in the scene.
        visualize_results: Visualize the spatio-temporal analysis results.

    Examples:
        >>> analyzer = SpatioTemporalAnalyzer()
        >>> frame = cv2.imread("image.jpg")
        >>> results = analyzer.process(frame, frame_number=1)
        >>> cv2.imshow("Spatio-Temporal Analysis", results.plot_im)
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize SpatioTemporalAnalyzer with parameters for spatio-temporal analysis."""
        super().__init__(**kwargs)

        self.history_length = self.CFG.get("history_length", 30)
        self.speed_threshold = self.CFG.get("speed_threshold", 5.0)  # pixels per frame
        self.interaction_distance = self.CFG.get("interaction_distance", 50.0)  # pixels
        self.prediction_horizon = self.CFG.get("prediction_horizon", 5)  # frames

        self.track_history: Dict[int, List[Tuple[float, float, float]]] = {}
        self.speed_data: Dict[int, float] = {}
        self.congestion_areas: List[Dict] = []

    def process(self, im0: np.ndarray, frame_number: int) -> SolutionResults:
        """
        Process image data and perform spatio-temporal analysis on object tracks.

        Args:
            im0 (np.ndarray): Input image for processing.
            frame_number (int): Current frame number for temporal analysis.

        Returns:
            (SolutionResults): Contains processed image `plot_im`, 'tracks' (dict, track information),
                'speeds' (dict, speed of each track), 'interactions' (list, detected interactions),
                and 'congestion' (list, congestion areas).

        Examples:
            >>> analyzer = SpatioTemporalAnalyzer()
            >>> frame = np.zeros((480, 640, 3), dtype=np.uint8)
            >>> results = analyzer.process(frame, frame_number=1)
        """
        self.extract_tracks(im0)

        # Update track history
        self.update_track_history(frame_number)

        # Estimate speed
        self.estimate_speed()

        # Predict future paths
        predictions = self.predict_path()

        # Detect interactions
        interactions = self.detect_interactions()

        # Analyze congestion
        self.analyze_congestion()

        # Visualize results
        plot_im = self.visualize_results(im0, predictions, interactions)

        # Return output dictionary with analysis results
        return SolutionResults(
            plot_im=plot_im,
            tracks=self.track_history,
            speeds=self.speed_data,
            interactions=interactions,
            congestion=self.congestion_areas
        )

    def update_track_history(self, frame_number: int) -> None:
        """
        Update the history of track positions with current frame data.

        Args:
            frame_number (int): Current frame number for temporal analysis.

        Examples:
            >>> analyzer = SpatioTemporalAnalyzer()
            >>> analyzer.boxes = np.array([[100, 100, 200, 200]])
            >>> analyzer.track_ids = np.array([1])
            >>> analyzer.update_track_history(1)
        """
        for i, (x1, y1, x2, y2, conf, cls) in enumerate(self.boxes):
            track_id = int(self.track_ids[i])
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            if track_id not in self.track_history:
                self.track_history[track_id] = []

            # Add current position and timestamp (frame number)
            self.track_history[track_id].append((center_x, center_y, frame_number))

            # Keep only the most recent history_length positions
            if len(self.track_history[track_id]) > self.history_length:
                self.track_history[track_id] = self.track_history[track_id][-self.history_length:]

    def estimate_speed(self) -> None:
        """
        Estimate the speed of each track based on position history.

        Examples:
            >>> analyzer = SpatioTemporalAnalyzer()
            >>> analyzer.track_history = {1: [(100, 100, 1), (110, 105, 2), (120, 110, 3)]}
            >>> analyzer.estimate_speed()
            >>> print(analyzer.speed_data[1])  # Should be around 7.07 pixels per frame
        """
        for track_id, history in self.track_history.items():
            if len(history) < 2:
                self.speed_data[track_id] = 0.0
                continue

            # Calculate speed based on the last two positions
            (x1, y1, t1), (x2, y2, t2) = history[-2], history[-1]
            distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            time_diff = max(t2 - t1, 1)  # Avoid division by zero
            speed = distance / time_diff

            self.speed_data[track_id] = speed

    def predict_path(self) -> Dict[int, List[Tuple[float, float]]]:
        """
        Predict the future path of each track using linear extrapolation.

        Returns:
            (Dict[int, List[Tuple[float, float]]]): Predicted future positions for each track.

        Examples:
            >>> analyzer = SpatioTemporalAnalyzer()
            >>> analyzer.track_history = {1: [(100, 100, 1), (110, 105, 2), (120, 110, 3)]}
            >>> predictions = analyzer.predict_path()
            >>> print(predictions[1])  # Should show predicted positions
        """
        predictions: Dict[int, List[Tuple[float, float]]] = {}

        for track_id, history in self.track_history.items():
            if len(history) < 2:
                predictions[track_id] = []
                continue

            # Use the last two positions to calculate direction and speed
            (x1, y1, t1), (x2, y2, t2) = history[-2], history[-1]
            direction_x = x2 - x1
            direction_y = y2 - y1
            time_diff = max(t2 - t1, 1)

            # Calculate velocity vector
            velocity_x = direction_x / time_diff
            velocity_y = direction_y / time_diff

            # Predict future positions
            predicted = []
            current_x, current_y = x2, y2
            for i in range(1, self.prediction_horizon + 1):
                next_x = current_x + velocity_x * i
                next_y = current_y + velocity_y * i
                predicted.append((next_x, next_y))

            predictions[track_id] = predicted

        return predictions

    def detect_interactions(self) -> List[Tuple[int, int, float]]:
        """
        Detect interactions between objects based on proximity.

        Returns:
            (List[Tuple[int, int, float]]): List of interacting track pairs with distance.

        Examples:
            >>> analyzer = SpatioTemporalAnalyzer()
            >>> analyzer.track_history = {
            ...     1: [(100, 100, 3)],
            ...     2: [(110, 105, 3)],
            ...     3: [(200, 200, 3)]
            ... }
            >>> interactions = analyzer.detect_interactions()
            >>> print(interactions)  # Should detect interaction between 1 and 2
        """
        interactions = []

        # Get current positions of all tracks
        current_positions = {}
        for track_id, history in self.track_history.items():
            if history:
                x, y, _ = history[-1]
                current_positions[track_id] = (x, y)

        # Check all pairs of tracks for proximity
        track_ids = list(current_positions.keys())
        for i in range(len(track_ids)):
            for j in range(i + 1, len(track_ids)):
                track1 = track_ids[i]
                track2 = track_ids[j]
                pos1 = current_positions[track1]
                pos2 = current_positions[track2]

                distance = np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
                if distance < self.interaction_distance:
                    interactions.append((track1, track2, distance))

        return interactions

    def analyze_congestion(self) -> None:
        """
        Analyze congestion in the scene based on object density.

        Examples:
            >>> analyzer = SpatioTemporalAnalyzer()
            >>> analyzer.boxes = np.array([
            ...     [100, 100, 150, 150],
            ...     [120, 120, 170, 170],
            ...     [140, 140, 190, 190]
            ... ])
            >>> analyzer.analyze_congestion()
            >>> print(analyzer.congestion_areas)  # Should detect a congestion area
        """
        if len(self.boxes) < 3:  # Need at least 3 objects to consider congestion
            self.congestion_areas = []
            return

        # Get center points of all objects
        centers = []
        for x1, y1, x2, y2, conf, cls in self.boxes:
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            centers.append((center_x, center_y))

        # Use DBSCAN to cluster objects and identify congestion areas
        if len(centers) >= 3:
            # Simple density-based approach: check distance between all pairs
            distances = spatial.distance.pdist(centers)
            avg_distance = np.mean(distances) if distances.size > 0 else float('inf')

            # If average distance is below threshold, consider it a congestion area
            congestion_threshold = 50.0  # pixels
            if avg_distance < congestion_threshold:
                # Calculate bounding box of congestion area
                xs = [x for x, y in centers]
                ys = [y for x, y in centers]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)

                self.congestion_areas = [{
                    "bbox": (x_min, y_min, x_max, y_max),
                    "density": len(centers) / ((x_max - x_min) * (y_max - y_min) + 1),
                    "count": len(centers)
                }]
            else:
                self.congestion_areas = []
        else:
            self.congestion_areas = []

    def visualize_results(self, im0: np.ndarray, predictions: Dict[int, List[Tuple[float, float]]],
                         interactions: List[Tuple[int, int, float]]) -> np.ndarray:
        """
        Visualize the spatio-temporal analysis results on the input image.

        Args:
            im0 (np.ndarray): Input image to visualize on.
            predictions (Dict[int, List[Tuple[float, float]]]): Predicted future positions.
            interactions (List[Tuple[int, int, float]]): Detected interactions.

        Returns:
            (np.ndarray): Image with visualization overlay.

        Examples:
            >>> analyzer = SpatioTemporalAnalyzer()
            >>> frame = np.zeros((480, 640, 3), dtype=np.uint8)
            >>> predictions = {1: [(130, 115), (140, 120), (150, 125)]}
            >>> interactions = [(1, 2, 25.5)]
            >>> result = analyzer.visualize_results(frame, predictions, interactions)
        """
        # Create a copy of the input image
        plot_im = im0.copy()

        # Draw track histories
        for track_id, history in self.track_history.items():
            if len(history) < 2:
                continue

            # Draw trajectory line
            points = [(int(x), int(y)) for x, y, _ in history]
            for i in range(1, len(points)):
                cv2.line(plot_im, points[i-1], points[i], (0, 255, 0), 2)

            # Draw current position
            current_x, current_y, _ = history[-1]
            cv2.circle(plot_im, (int(current_x), int(current_y)), 5, (0, 255, 0), -1)
            cv2.putText(plot_im, f"ID: {track_id}", (int(current_x) + 10, int(current_y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw speed information
            if track_id in self.speed_data:
                speed = self.speed_data[track_id]
                color = (0, 0, 255) if speed > self.speed_threshold else (0, 255, 0)
                cv2.putText(plot_im, f"Speed: {speed:.1f}", (int(current_x) + 10, int(current_y) + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw predicted paths
        for track_id, path in predictions.items():
            if path:
                current_x, current_y, _ = self.track_history[track_id][-1]
                # Draw predicted path as dotted line
                for i, (px, py) in enumerate(path):
                    cv2.circle(plot_im, (int(px), int(py)), 3, (255, 0, 0), -1)
                    if i > 0:
                        prev_px, prev_py = path[i-1]
                        cv2.line(plot_im, (int(prev_px), int(prev_py)), (int(px), int(py)), (255, 0, 0), 1, cv2.LINE_AA)

        # Draw interactions
        for track1, track2, distance in interactions:
            if track1 in self.track_history and track2 in self.track_history:
                x1, y1, _ = self.track_history[track1][-1]
                x2, y2, _ = self.track_history[track2][-1]
                cv2.line(plot_im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
                cv2.putText(plot_im, f"Dist: {distance:.1f}", (int((x1 + x2) / 2), int((y1 + y2) / 2) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Draw congestion areas
        for area in self.congestion_areas:
            x_min, y_min, x_max, y_max = area["bbox"]
            cv2.rectangle(plot_im, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)
            cv2.putText(plot_im, f"Congestion: {area['count']} objects", (int(x_min), int(y_min) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return plot_im
