import os
import cv2
import torch
import numpy as np
from models.face_vid2vid.deep_avatar import DeepAvatar
from skimage.transform import resize

cpu_mode = False

deep_avatar = DeepAvatar(avatar="6.png")
source = torch.tensor(deep_avatar.avatar_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)

if not cpu_mode:
    source = source.cuda()

cap = cv2.VideoCapture(0)

while cv2.waitKey(33) < 0:
    ret, frame = cap.read()
    if ret:
        frame = frame[112:-112, 192:-192] # crop center(256, 256)
        frame = resize(frame, (256, 256))[..., :3]
        result = deep_avatar.get_action_from_frame(frame=frame, source=source, ypr=(0, 0, 0))
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        #frame = cv2.convertScaleAbs(frame, alpha=255.0)
        cv2.imshow('frame', frame)
        cv2.imshow('result', result)

cap.release()
cv2.destroyAllWindows()
