import cv2
import numpy as np
import os
import mediapipe as mp
import glob
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_styled_landmarks(image, results):
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             )


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    return np.concatenate([pose,])


DATA_PATH = os.path.join('MP_Data')
actions = np.array(['666', '888', ])

no_sequences = 2
sequence_length = 30
for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass


# Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # 遍历每个动作
    for action in actions:
        file_list = glob.glob('./1Intercept_Data/' + action + '/*')
        # Loop through sequences aka videos
        for sequence in range(no_sequences):
            video_path=file_list[sequence]
            cap = cv2.VideoCapture(video_path)
            # 检查摄像头是否成功打开
            if not cap.isOpened():
                print("Error: Couldn't open the video.")
                continue
            frame_num=0
            while True:
                # Read feed
                ret, frame = cap.read()
                # 检查是否成功读取帧
                if not ret:
                    break  # 如果读取失败（通常是视频结束），则退出循环
                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)
                cv2.imshow('OpenCV Feed', image)

                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)
                frame_num=frame_num+1
                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
cap.release()
cv2.destroyAllWindows()
