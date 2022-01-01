import cv2
import torch
import numpy as np
from moviepy.editor import *
from skimage.transform import resize

from models.face_vid2vid.utils import load_checkpoints, set_driving, make_animation


class DeepAvatar:
    def __init__(self, image_path: str = './asset/avatar/', **kwargs: dict) -> None:
        super(DeepAvatar, self).__init__()
        self.model_name: str = 'DeepAvatar'
        self.model_version: str = '1.0.0'
        self.model_path: str = './models/face_vid2vid'
        self.driving = None
        self.kp_canonical = None
        self.kp_source = None
        self.kp_driving_initial = None

        # load pretrained model
        self.generator, self.kp_detector, self.he_estimator = self.load_pretrained_model()

        # get avatar
        self.avatar_name: str = kwargs['avatar']
        avatar_image_path: str = os.path.join(image_path, self.avatar_name)
        avatar_image = cv2.imread(avatar_image_path)
        avatar_image = cv2.cvtColor(avatar_image, cv2.COLOR_BGR2RGB)
        self.avatar_image = resize(avatar_image, (256, 256))[..., :3]

    def load_pretrained_model(self):
        return load_checkpoints(config_path=f'{self.model_path}/config/vox-256-spade.yaml',
                                checkpoint_path=f'{self.model_path}/ckpt/00000189-checkpoint.pth.tar',
                                gen='spade',
                                cpu_mode=False)

    def get_action_from_frame(self, frame, source, ypr: tuple = (0, 0, 0)) -> (list, int):
        # parameter parsing
        yaw = ypr[0]
        pitch = ypr[1]
        roll = ypr[2]

        # Resizing Frame Needed

        options_dict: dict = {
            'avatar_image': self.avatar_image,
            'fra': frame,
            'generator': self.generator,
            'kp_detector': self.kp_detector,
            'he_estimator': self.he_estimator,
            'relative': True,
            'adapt_movement_scale': True,
            'estimate_jacobian': False,
            'cpu_mode': False,
            'free_view': False,
            'yaw': yaw,
            'pitch': pitch,
            'roll': roll
        }

        if self.driving is None:
            self.driving = torch.tensor(frame[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
            self.kp_canonical, self.kp_source, self.kp_driving_initial = set_driving(self.driving, self.kp_detector, self.he_estimator, source)

        return make_animation(frame, source, self.kp_canonical, self.kp_source, self.kp_driving_initial, options_dict)
