import cv2
import os
import glob
import numpy as np
# 设置保存图像的路径
save_path = './1Intercept_Data/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
# file_list=glob.glob('./images/'+style+'/*/*')
# actions = np.array(['666', 'thumbs_up', 'finger_heart','scissor_hand'])
actions = np.array(['666', '888'])
for action in actions:
    file_list=glob.glob('./0Initial_Data/'+action+'/*')
    #print(file_list)
    for video_path in file_list:
        cap = cv2.VideoCapture(video_path)
        # 检查视频是否成功打开
        if not cap.isOpened():
            print(f"Error: Couldn't open the video file {video_path}.")
            continue
        # 初始化帧计数器
        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4编码

        # 创建子文件夹以视频文件名命名
        video_name = video_path.split('\\')[1]
        # print(video_name)
        for_save_path = save_path+action+'\\'
        if not os.path.exists(for_save_path):
            os.makedirs(for_save_path)
        sub_save_path=for_save_path+video_name
        out = cv2.VideoWriter(sub_save_path, fourcc, fps,(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        while frame_count < 30 and cap.isOpened():
            # 读取下一帧
            ret, frame = cap.read()
            # 如果成功读取帧
            if ret:
                out.write(frame)
                # 增加帧计数器
                frame_count += 1
            else:
                # 如果读取失败，通常是因为到达视频末尾
                break
        # 释放视频捕获对象
        cap.release()

# 关闭所有OpenCV窗口
cv2.destroyAllWindows()