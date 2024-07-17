import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import numpy as np
import cv2

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

BaseOptions = mp.tasks.BaseOptions #reference to the class not the instance of the class
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode
model_path = '/home/rohan/fitnessTrainer/pose_landmarker_full.task'
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE)

with PoseLandmarker.create_from_options(options) as landmarker:
    mp_image = mp.Image.create_from_file('/home/rohan/fitnessTrainer/20221225_202104.jpg')
    pose_landmarker_result = landmarker.detect(mp_image)
    print(type(pose_landmarker_result))
    annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), pose_landmarker_result)
    cv2.imshow('Image',cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)) 
    cv2.waitKey(0)   
  
