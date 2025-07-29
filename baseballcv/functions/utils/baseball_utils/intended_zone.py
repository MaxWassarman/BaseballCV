import logging
import cv2
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
import pandas as pd
from ultralytics import YOLO
import io
from contextlib import redirect_stdout
from baseballcv.functions.load_tools import LoadTools
import math
import mediapipe as mp
from baseballcv.utilities import BaseballCVLogger, ProgressBar
import torch
import urllib.request

class IntendedZone:
    """
    Class for calculating and analyzing intended target zones using catcher positioning.
    
    This class uses computer vision models to detect the catcher, glove, home plate, and 
    pitching rubber, then calculates the intended target's lateral position relative to home plate.
    """

    def __init__(
        self, 
        catcher_model: str = 'phc_detector',
        glove_model: str = 'glove_tracking',
        homeplate_model: str = 'glove_tracking',
        results_dir: str = "intended_zone_results",
        verbose: bool = True,
        device: str = None,
        catcher_depth: float = 5.0,
        logger: logging.Logger = None
    ):
        """
        Initialize the IntendedZone class.
        
        Args:
            catcher_model (str): Model name for detecting catchers
            glove_model (str): Model name for detecting gloves
            homeplate_model (str): Model name for detecting home plate
            results_dir (str): Directory to save results
            verbose (bool): Whether to print detailed progress information
            device (str): Device to run models on (cpu, cuda, etc.)
            catcher_depth (float): Depth of catcher behind home plate in feet
            logger (logging.Logger): Logger instance for logging
        """
        self.load_tools = LoadTools()
        # Had to change to txt files based. The load tools doesn't work for some reason for the phc_detector.
        self.catcher_model = YOLO(r'C:\Users\maxwa\baseball\BaseballCV\models\od\YOLO\pitcher_hitter_catcher_detector\model_weights\pitcher_hitter_catcher_detector_v3.pt')
        self.glove_model = YOLO(r'C:\Users\maxwa\baseball\BaseballCV\models\od\YOLO\glove_tracking\model_weights\glove_tracking_v4_YOLOv11.pt')
        self.homeplate_model = YOLO(r'C:\Users\maxwa\baseball\BaseballCV\models\od\YOLO\glove_tracking\model_weights\glove_tracking_v4_YOLOv11.pt')

        self.logger = logger if logger is not None else BaseballCVLogger.get_logger(__name__)
        
        if verbose:
            self.logger.info(f"Models loaded: {catcher_model}, {glove_model}, {homeplate_model}")

        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

        self.verbose = verbose
        self.device = device
        self.catcher_depth = catcher_depth

        # Initialize MediaPipe pose for release point detection
        self.mp_pose = mp.solutions.pose
        
        # Download MediaPipe pose model if needed
        self._setup_pose_detector()

    def _setup_pose_detector(self):
        """Download and setup MediaPipe pose detector."""
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
        filename = "pose_landmarker.task"
        
        if not os.path.exists(filename):
            if self.verbose:
                self.logger.info("Downloading pose model")
            urllib.request.urlretrieve(url, filename)
            if self.verbose:
                self.logger.info("Download done")
        
        # Create MediaPipe detector
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        
        # Turn off MediaPipe's non-error logging
        import absl.logging
        absl.logging.set_verbosity(absl.logging.ERROR)
        
        base_options = python.BaseOptions(model_asset_path=filename)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=True,
            min_pose_detection_confidence=0.3,
            min_tracking_confidence=0.3,
            min_pose_presence_confidence=0.3
        )
        self.pose_detector = vision.PoseLandmarker.create_from_options(options)

    def analyze(
        self,
        start_date: str,
        end_date: str,
        team_abbr: str = None,
        player: int = None,
        pitch_type: str = None,
        max_videos: int = None,
        max_videos_per_game: int = None,
        save_csv: bool = True,
        csv_path: str = None
    ) -> List[Dict]:
        """
        Analyze videos from a date range to calculate intended target location.
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            team_abbr (str): Team abbreviation to filter by
            player (int): Player ID to filter by
            pitch_type (str): Pitch type to filter by (e.g., "FF")
            max_videos (int): Maximum number of videos to process
            max_videos_per_game (int): Maximum videos per game
            save_csv (bool): Whether to save analysis results to CSV
            csv_path (str): Custom path for CSV file
            
        Returns:
            List[Dict]: List of analysis results per video
        """
        from baseballcv.functions.savant_scraper import BaseballSavVideoScraper
        
        savant_scraper = BaseballSavVideoScraper.from_date_range(
            start_dt=start_date, 
            end_dt=end_date,
            player=player,
            team_abbr=team_abbr, 
            pitch_type=pitch_type,
            max_return_videos=max_videos, 
            max_videos_per_game=max_videos_per_game,
            download_folder="savant_videos"
        )

        download_folder = "savant_videos"
        
        savant_scraper.run_executor()
        # Use the play_ids_df attribute directly instead of get_play_ids_df() method. Needed with new scraper updates.
        pitch_data = savant_scraper.play_ids_df
        video_files = [os.path.join(download_folder, f) for f in os.listdir(download_folder) if f.endswith('.mp4')]

        intended_zone_results = []
        detailed_results = []
       
        for video_path in video_files:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            play_id = video_name.split('_')[-1]
            game_pk = video_name.split('_')[-2]
        
            # Find corresponding pitch data
            pitch_data_row = None
            for _, row in pitch_data.iterrows():
                if row["play_id"] == play_id:
                    pitch_data_row = row
                    break
            
            if pitch_data_row is None:
                if self.verbose:
                    self.logger.info(f"No pitch data found for play_id {play_id}, skipping...")
                continue

            # Calculate intended zone
            intended_zone_data = self._video_intended_location(video_path)

            # Collect results
            results = {
                "video_name": video_name,
                "play_id": play_id,
                "game_pk": game_pk,
                "intended_zone_data": intended_zone_data
            }

            # Detailed data for CSV
            detailed_data = {
                "video_name": video_name,
                "play_id": play_id,
                "game_pk": game_pk,
                
                # Intended target data
                "plate_loc_side": intended_zone_data.get("PlateLocSide") if intended_zone_data else None,
                "plate_loc_height": intended_zone_data.get("PlateLocHeight") if intended_zone_data else None,
                "raw_x": intended_zone_data.get("rawX") if intended_zone_data else None,
                "raw_y": intended_zone_data.get("rawY") if intended_zone_data else None,
                "translated_x": intended_zone_data.get("translatedX") if intended_zone_data else None,
                "translated_y": intended_zone_data.get("translatedY") if intended_zone_data else None,
                "release_frame": intended_zone_data.get("release_frame") if intended_zone_data else None,
                
                # Success flags
                "calculation_successful": intended_zone_data is not None,
                
                # Catcher depth used
                "catcher_depth": self.catcher_depth
            }

            # Add Statcast data if available
            if pitch_data_row is not None:
                for key, value in pitch_data_row.items():
                    detailed_data[f"statcast_{key}"] = value
            
            detailed_results.append(detailed_data)
            intended_zone_results.append(results)
            
            if self.verbose:
                plate_loc_side = intended_zone_data.get("PlateLocSide") if intended_zone_data else "N/A"
                self.logger.info(f"Play {play_id}: PlateLocSide = {plate_loc_side}")
        
        # Save detailed data to CSV if requested
        if save_csv and detailed_results:
            if csv_path is None:
                csv_path = os.path.join(self.results_dir, "intended_zone_results.csv")
            
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
            
            # Create DataFrame from detailed results
            df = pd.DataFrame(detailed_results)
            
            # Save to CSV
            df.to_csv(csv_path, index=False)
            
            if self.verbose:
                self.logger.info(f"Saved detailed results to {csv_path}")
                self.logger.info(f"CSV contains {len(df)} rows with {len(df.columns)} columns of data")
        
        return intended_zone_results

    def _video_intended_location(self, video_path: str) -> Optional[Dict]:
        """
        Calculate intended location on video.
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            Optional[Dict]: Intended zone data or None if calculation fails
        """
        try:
            # Step 1: Estimate the release frame
            # Random though but could I use a ball model instead and then split to like every 5th frame and then just choose then one where the ball first appears?
            # This would likely speed everything up quite a bit
            release_frame = self._release_frame_estimation(video_path)
            if release_frame is None:
                if self.verbose:
                    self.logger.warning(f"Could not estimate release frame for {video_path}")
                return None

            # Step 2: Get the frame at release and calculate intended zone
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, release_frame)
            ret, static_frame = cap.read()
            cap.release()
            
            if not ret:
                if self.verbose:
                    self.logger.warning(f"Could not read frame {release_frame}")
                return None

            # Step 3: Calculate intended zone at the release frame
            intended_zone_result = self._intended_zone(
                static_frame, 
                self.catcher_model, 
                self.glove_model, 
                self.catcher_depth
            )
            
            if intended_zone_result:
                intended_zone_result["release_frame"] = release_frame
                
            return intended_zone_result

        except Exception as e:
            if self.verbose:
                self.logger.error(f"Error in video intended location calculation: {e}")
            return None

    def _release_frame_estimation(self, video_path: str) -> Optional[int]:
        """
        Estimate release frame using pitcher detection and pose estimation.
        Based on the reference code's release_frame_estimation function.
        """
        try:
            # Infer pitcher locations throughout the video
            with io.StringIO() as buf, redirect_stdout(buf):
                results_phc = self.catcher_model.predict(source=video_path, save=False, verbose=False)

            # Load video into cv2 and initialize storage variables
            cap = cv2.VideoCapture(video_path)
            
            # Initialize storage variables
            xyz_dict = {key: {"x": [], "y": [], "z": []} for key in ["lw", "rw", "ls", "rs"]}
            xyz_dict["frame"] = []
            
            frame_idx = 0
            # Loop through each result from PHC model
            for result in results_phc:
                # Get the index of the pitcher (class 1 in PHC model)
                pitcher_indices = torch.where(result.boxes.cls == 1)[0].tolist()
                
                # If there is no pitcher detected, move on
                if not pitcher_indices:
                    frame_idx += 1
                    continue
                else:
                    # Get pitcher box location
                    pitcher = result.boxes[pitcher_indices[0]]
                    x1, y1, x2, y2 = pitcher.xyxy[0]
                    
                    # Get the frame and crop the pitcher box
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if not ret:
                        frame_idx += 1
                        continue
                        
                    cropped_image = frame[int(y1):int(y2), int(x1):int(x2)]
                    if cropped_image.size == 0:
                        frame_idx += 1
                        continue
                        
                    cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

                    # Have MediaPipe evaluate just the pitcher box
                    mp_image_eval = mp.Image(
                        image_format=mp.ImageFormat.SRGB, 
                        data=cropped_image_rgb
                    )
                    detection_result = self.pose_detector.detect(mp_image_eval)

                    # If we don't detect any poses, skip it
                    if len(detection_result.pose_landmarks) == 0:
                        frame_idx += 1
                        continue
                    else:
                        # Store wrist and shoulder locations (like reference code)
                        landmarks = detection_result.pose_landmarks[0]
                        
                        xyz_dict["lw"]["z"].append(landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].z)
                        xyz_dict["rw"]["z"].append(landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST].z)
                        xyz_dict["ls"]["z"].append(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].z)
                        xyz_dict["rs"]["z"].append(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].z)
                        xyz_dict["lw"]["x"].append(landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].x)
                        xyz_dict["rw"]["x"].append(landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST].x)
                        xyz_dict["ls"]["x"].append(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x)
                        xyz_dict["rs"]["x"].append(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x)
                        xyz_dict["lw"]["y"].append(landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].y)
                        xyz_dict["rw"]["y"].append(landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST].y)
                        xyz_dict["ls"]["y"].append(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y)
                        xyz_dict["rs"]["y"].append(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y)
                        xyz_dict["frame"].append(frame_idx)
                    
                    frame_idx += 1

            cap.release()
            
            # Return the estimated release frame based on the results
            if len(xyz_dict["frame"]) > 5:
                return self._estimated_release_frame(xyz_dict)
            else:
                if self.verbose:
                    self.logger.warning("Insufficient pose data for release frame estimation")
                return None
                
        except Exception as e:
            if self.verbose:
                self.logger.error(f"Error in release frame estimation: {e}")
            return None

    def _estimated_release_frame(self, xyz_dict: Dict) -> int:
        """
        Estimate release frame from pose data.
        Based on the reference code's estimated_release_frame function.
        """
        # Get estimated throwing hand
        rw_range = max(xyz_dict["rw"]['z']) - min(xyz_dict["rw"]['z'])
        lw_range = max(xyz_dict["lw"]['z']) - min(xyz_dict["lw"]['z'])
        
        throw_hand = 'rw' if rw_range > lw_range else 'lw'
        glove_hand = 'lw' if throw_hand == 'rw' else 'rw'

        # What frame does the arm reach maximum distance toward camera
        max_index = xyz_dict[throw_hand]['z'].index(min(xyz_dict[throw_hand]['z']))

        # Get points around key frame, our release is probably in here
        search_start = max(0, max_index - 10)
        search_end = min(len(xyz_dict["frame"]), max_index + 10)
        
        local_x_lw = xyz_dict[glove_hand]['x'][search_start:search_end]
        local_x_rw = xyz_dict[throw_hand]["x"][search_start:search_end]
        local_frame = xyz_dict["frame"][search_start:search_end]

        # Get the max distance between wrists in x plane within this region
        max_distance = [abs(l - r) for l, r in zip(local_x_lw, local_x_rw)]

        # Assume release is the maximum distance between the wrists within the range we searched
        max_distance_index = max_distance.index(max(max_distance))

        return local_frame[max_distance_index]

    def _intended_zone(self, frame: np.ndarray, model_phc: YOLO, model_plate: YOLO, catcher_depth: float = 6) -> Optional[Dict]:
        """
        Calculate intended zone from a frame.
        Based on the reference code's intended_zone function.
        """
        try:
            with io.StringIO() as buf, redirect_stdout(buf):
                results_phc = model_phc.predict(source=frame, save=False, verbose=False)
                results_plate = model_plate.predict(source=frame, save=False, verbose=False)

            # Extract required objects
            # Class indices: catcher=2, plate=1, rubber=3, glove=0
            try:
                catcher = results_phc[0].boxes[torch.where(results_phc[0].boxes.cls == 2.0)[0].tolist()[0]]
                plate = results_plate[0].boxes[torch.where(results_plate[0].boxes.cls == 1.0)[0].tolist()[0]]
                rubber = results_plate[0].boxes[torch.where(results_plate[0].boxes.cls == 3.0)[0].tolist()[0]]
                glove = results_plate[0].boxes[torch.where(results_plate[0].boxes.cls == 0.0)[0].tolist()[0]]
            except (IndexError, AttributeError):
                if self.verbose:
                    self.logger.warning("Could not detect all required objects (catcher, plate, rubber, glove)")
                return None

            # Extract coordinates
            cx1, cy1, cx2, cy2 = catcher.xyxy[0]
            px1, py1, px2, py2 = plate.xyxy[0]
            rx1, ry1, rx2, ry2 = rubber.xyxy[0]
            gx1, gy1, gx2, gy2 = glove.xyxy[0]
            depth = abs(int(rx1) - int(rx2)) / 2 * 5 / 6

            # Calculate intended zone pixels
            izx, izy = self._intended_zone_pixels(
                x=[int(cx1), int(cx2), int(gx1), int(gx2)],
                y=[int(cy1), int(cy2), int(gy1), int(gy2)]
            )

            # Calculate ground line slope
            ground_line_slope = self._slope_of_line(
                (int(np.mean([int(px1), int(px2)])), int(np.mean([int(py1), int(py2)]))),
                (
                    int(np.mean([int(rx1), int(rx2)])),
                    int(np.mean([int(ry1), int(ry2)])) + int(depth),
                ),
            )

            # Project catcher position to home plate
            catcher_beyond_pixels = (
                self._distance_between_points(
                    (int(np.mean([int(px1), int(px2)])), int(np.mean([int(py1), int(py2)]))),
                    (
                        int(np.mean([int(rx1), int(rx2)])),
                        int(np.mean([int(ry1), int(ry2)])) + int(depth),
                    ),
                )
                / 59.5
                * catcher_depth
            )
            
            translated = self._find_endpoint(
                (izx, izy), ground_line_slope, -1 * catcher_beyond_pixels
            )
            
            # Convert to real coordinates
            real_coords = self._real_coordinates(float(translated[0]), float(izy), plate)

            # Return results
            return {
                "rawX": float(izx),
                "rawY": float(izy), 
                "translatedX": float(translated[0]),
                "translatedY": float(translated[1]),
                "PlateLocSide": float(real_coords[0]),
                "PlateLocHeight": float(real_coords[1])
            }

        except Exception as e:
            if self.verbose:
                self.logger.error(f"Error in intended zone calculation: {e}")
            return None

    # Helper functions
    def _slope_of_line(self, point1: Tuple, point2: Tuple) -> Optional[float]:
        """Calculate slope between two points."""
        x1, y1 = point1
        x2, y2 = point2
        if x1 == x2:
            return None
        return (y2 - y1) / (x2 - x1)

    def _find_endpoint(self, start_point: Tuple, slope: Optional[float], length: float) -> Tuple:
        """Find endpoint along a line with given slope and length."""
        x1, y1 = start_point
        if slope is None:
            # Vertical line case
            return (x1, y1 + length)
        
        angle = math.atan(slope)
        delta_x = math.cos(angle) * length
        delta_y = math.sin(angle) * length
        return (x1 + delta_x, y1 + delta_y)

    def _distance_between_points(self, point1: Tuple, point2: Tuple) -> float:
        """Calculate Euclidean distance between two points."""
        x1, y1 = point1
        x2, y2 = point2
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def _intended_zone_pixels(self, x: List[int], y: List[int], 
                            wx: List[float] = None, wy: List[float] = None) -> Tuple[float, float]:
        """
        Calculate weighted average of intended zone positioning coordinates.
        Default weights from reference code: glove is triple weighted in X, catcher top is double weighted in Y.
        """
        if wx is None:
            wx = [1, 1, 3, 3]  # Triple weight for glove in x
        if wy is None:
            wy = [2, 1, 1, 1]  # Double weight for catcher top in y
        
        return (np.average(x, weights=wx), np.average(y, weights=wy))

    def _real_coordinates(self, izx: float, izy: float, plate) -> Tuple[float, float]:
        """Convert pixel coordinates to real-world inches relative to home plate."""
        px1, py1, px2, py2 = plate.xyxy[0]
        px1, py1, px2, py2 = float(px1), float(py1), float(px2), float(py2)

        pw = abs(px1 - px2)
        height = abs(((izy - max([py1, py2])) * 17 / pw))
        side = ((izx - np.mean([px1, px2])) * 17 / pw)
        return (side, height)