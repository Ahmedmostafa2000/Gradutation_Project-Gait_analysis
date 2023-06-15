import cv2
import mediapipe as mp
import pandas as pd
import os
import time
#initiating mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_drawing_styles = mp.solutions.drawing_styles


video_count = 0
for i in os.listdir("D:/Career/GradProj/Datasets/Normal"):
  video_count+=1
  #initiating constants
  FILE_NAME = "D:/Career/GradProj/Datasets/Normal/"+i
  # VIDEO_NAME = "1_out"
  N_TPOINTS = 6



  #initiating video caputure
  cap = cv2.VideoCapture(FILE_NAME)



  # out = cv2.VideoWriter(VIDEO_NAME+".mp4",
  # 		cv2.VideoWriter_fourcc(*"MJPG"),
  # 		fps = 60,frameSize=(1920,1080))

  #initiating records
  pose_lms = [0]*32
  record = []
  frame = 0
  with mp_pose.Pose(
      min_detection_confidence=0.5,
      min_tracking_confidence=0.4) as pose:
    while cap.isOpened():
      success, image = cap.read()
      try:
        image = cv2.resize(image, (160*8,120*8))
      except:
        break
      frame+=1
      if not success:
        break
      # Flip the image horizontally for a later selfie-view display, and convert
      # the BGR image to RGB.
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      results = pose.process(image)
      landmark_list = []
      pose_lm = results.pose_landmarks
      if pose_lm:
        for j in range(len(pose_lm.landmark)):
          landmark_list.append([pose_lm.landmark[j].x,pose_lm.landmark[j].y,pose_lm.landmark[j].z])
      mp_drawing.draw_landmarks(image,
                                results.pose_landmarks,
                                mp_pose.POSE_CONNECTIONS,
                                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

      if pose_lm:
        record.append(landmark_list)
      # Draw the pose annotation on the image.
      # image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      mp_drawing.draw_landmarks(
          image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
      
      
      # cv2.imshow('MediaPipe Pose', image)
      if cv2.waitKey(1) & 0xFF == ord("q"):
        break
      
  cap.release()
  # out.release()

  df = pd.DataFrame(record)
  df.to_csv("D:\\Career\\GradProj\\Datasets\\Normal_csvFiles\\"+i+".csv")
  print("%.4f       "%(100*video_count/300),end = '\r')