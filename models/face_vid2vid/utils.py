import matplotlib

matplotlib.use('Agg')
import cv2
import sys
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from models.face_vid2vid.sync_batchnorm import DataParallelWithCallback

from models.face_vid2vid.modules.generator import OcclusionAwareGenerator, OcclusionAwareSPADEGenerator
from models.face_vid2vid.modules.keypoint_detector import KPDetector, HEEstimator
from models.face_vid2vid.animate import normalize_kp

if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")


def load_checkpoints(config_path, checkpoint_path, gen, cpu_mode=False):
    with open(config_path) as f:
        config = yaml.load(f)

    if gen == 'original':
        generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                            **config['model_params']['common_params'])
    elif gen == 'spade':
        generator = OcclusionAwareSPADEGenerator(**config['model_params']['generator_params'],
                                                 **config['model_params']['common_params'])

    if not cpu_mode:
        generator.cuda()

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if not cpu_mode:
        kp_detector.cuda()

    he_estimator = HEEstimator(**config['model_params']['he_estimator_params'],
                               **config['model_params']['common_params'])
    if not cpu_mode:
        he_estimator.cuda()

    if cpu_mode:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)

    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])
    he_estimator.load_state_dict(checkpoint['he_estimator'])

    if not cpu_mode:
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)
        he_estimator = DataParallelWithCallback(he_estimator)

    generator.eval()
    kp_detector.eval()
    he_estimator.eval()

    return generator, kp_detector, he_estimator


def headpose_pred_to_degree(pred):
    device = pred.device
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).to(device)
    pred = F.softmax(pred, dim=1)
    return torch.sum(pred * idx_tensor, axis=1) * 3 - 99


def get_rotation_matrix(yaw, pitch, roll):
    yaw = yaw / 180 * 3.14
    pitch = pitch / 180 * 3.14
    roll = roll / 180 * 3.14

    roll = roll.unsqueeze(1)
    pitch = pitch.unsqueeze(1)
    yaw = yaw.unsqueeze(1)

    pitch_mat = torch.cat([torch.ones_like(pitch), torch.zeros_like(pitch), torch.zeros_like(pitch),
                           torch.zeros_like(pitch), torch.cos(pitch), -torch.sin(pitch),
                           torch.zeros_like(pitch), torch.sin(pitch), torch.cos(pitch)], dim=1)
    pitch_mat = pitch_mat.view(pitch_mat.shape[0], 3, 3)

    yaw_mat = torch.cat([torch.cos(yaw), torch.zeros_like(yaw), torch.sin(yaw),
                         torch.zeros_like(yaw), torch.ones_like(yaw), torch.zeros_like(yaw),
                         -torch.sin(yaw), torch.zeros_like(yaw), torch.cos(yaw)], dim=1)
    yaw_mat = yaw_mat.view(yaw_mat.shape[0], 3, 3)

    roll_mat = torch.cat([torch.cos(roll), -torch.sin(roll), torch.zeros_like(roll),
                          torch.sin(roll), torch.cos(roll), torch.zeros_like(roll),
                          torch.zeros_like(roll), torch.zeros_like(roll), torch.ones_like(roll)], dim=1)
    roll_mat = roll_mat.view(roll_mat.shape[0], 3, 3)

    return torch.einsum('bij,bjk,bkm->bim', pitch_mat, yaw_mat, roll_mat)

def set_driving(driving, kp_detector_model, he_estimator_model, source):
    kp_canonical = kp_detector_model(source)
    he_source = he_estimator_model(source)
    he_driving_initial = he_estimator_model(driving)

    kp_source = keypoint_transformation(kp_canonical, he_source, False)
    kp_driving_initial = keypoint_transformation(kp_canonical, he_driving_initial, False)

    return kp_canonical, kp_source, kp_driving_initial


def keypoint_transformation(kp_canonical, he, estimate_jacobian=True, free_view=False, yaw=0, pitch=0, roll=0):
    kp = kp_canonical['value']
    if not free_view:
        yaw, pitch, roll = he['yaw'], he['pitch'], he['roll']
        yaw = headpose_pred_to_degree(yaw)
        pitch = headpose_pred_to_degree(pitch)
        roll = headpose_pred_to_degree(roll)
    else:
        if yaw is not None:
            yaw = torch.tensor([yaw]).cuda()
        else:
            yaw = he['yaw']
            yaw = headpose_pred_to_degree(yaw)
        if pitch is not None:
            pitch = torch.tensor([pitch]).cuda()
        else:
            pitch = he['pitch']
            pitch = headpose_pred_to_degree(pitch)
        if roll is not None:
            roll = torch.tensor([roll]).cuda()
        else:
            roll = he['roll']
            roll = headpose_pred_to_degree(roll)

    t, exp = he['t'], he['exp']

    rot_mat = get_rotation_matrix(yaw, pitch, roll)

    # keypoint rotation
    kp_rotated = torch.einsum('bmp,bkp->bkm', rot_mat, kp)

    # keypoint translation
    t = t.unsqueeze_(1).repeat(1, kp.shape[1], 1)
    kp_t = kp_rotated + t

    # add expression deviation
    exp = exp.view(exp.shape[0], -1, 3)
    kp_transformed = kp_t + exp

    if estimate_jacobian:
        jacobian = kp_canonical['jacobian']
        jacobian_transformed = torch.einsum('bmp,bkps->bkms', rot_mat, jacobian)
    else:
        jacobian_transformed = None

    return {'value': kp_transformed, 'jacobian': jacobian_transformed}

def make_animation(frame, source, kp_canonical, kp_source, kp_driving_initial, options_dict):
    avatar_image = options_dict['avatar_image']
    generator_model = options_dict['generator']
    kp_detector_model = options_dict['kp_detector']
    he_estimator_model = options_dict['he_estimator']
    relative = options_dict['relative']
    adapt_movement_scale = options_dict['adapt_movement_scale']
    jacobian = options_dict['estimate_jacobian']
    cpu_mode = options_dict['cpu_mode']
    free_view = options_dict['free_view']
    yaw = options_dict['yaw']
    pitch = options_dict['pitch']
    roll = options_dict['roll']

    result_frames = []

    driving_frame = torch.tensor(frame[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)

    if not cpu_mode:
        driving_frame = driving_frame.cuda()
    he_driving = he_estimator_model(driving_frame)
    kp_driving = keypoint_transformation(kp_canonical, he_driving, jacobian, free_view=free_view,
                                         yaw=yaw, pitch=pitch, roll=roll)
    kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                           kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                           use_relative_jacobian=jacobian, adapt_movement_scale=adapt_movement_scale)
    out = generator_model(source, kp_source=kp_source, kp_driving=kp_norm)

    result_frame = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]

    return result_frame
