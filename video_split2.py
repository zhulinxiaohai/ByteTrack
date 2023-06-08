# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import cv2
import mmcv
import numpy as np
import os
import os.path as osp
import shutil
import torch
import warnings
from scipy.optimize import linear_sum_assignment

# from pyskl.apis import inference_recognizer, init_recognizer

try:
    from mmdet.apis import inference_detector, init_detector
except (ImportError, ModuleNotFoundError):
    def inference_detector(*args, **kwargs):
        pass

    def init_detector(*args, **kwargs):
        pass
    warnings.warn(
        'Failed to import `inference_detector` and `init_detector` from `mmdet.apis`. '
        'Make sure you can successfully import these if you want to use related features. '
    )

try:
    from mmpose.apis import inference_top_down_pose_model, init_pose_model, vis_pose_result
except (ImportError, ModuleNotFoundError):
    def init_pose_model(*args, **kwargs):
        pass

    def inference_top_down_pose_model(*args, **kwargs):
        pass

    def vis_pose_result(*args, **kwargs):
        pass

    warnings.warn(
        'Failed to import `init_pose_model`, `inference_top_down_pose_model`, `vis_pose_result` from '
        '`mmpose.apis`. Make sure you can successfully import these if you want to use related features. '
    )


try:
    import moviepy.editor as mpy
except ImportError:
    raise ImportError('Please install moviepy to enable output file')

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.75
FONTCOLOR = (255, 255, 255)  # BGR, white
THICKNESS = 1
LINETYPE = 1


def parse_args():
    parser = argparse.ArgumentParser(description='PoseC3D demo')
    # parser.add_argument('--video', default='split/hiv00047.mp4', help='video file/url')
    parser.add_argument('--video', default='data/64/hiv00020.mp4', help='video file/url')
    parser.add_argument('--out_filename', default='split/out/', help='output filename')
    parser.add_argument(
        '--config',
        default='configs/posec3d/slowonly_r50_ntu120_xsub/joint.py',
        help='skeleton action recognition config file path')
    parser.add_argument(
        '--checkpoint',
        default='https://download.openmmlab.com/mmaction/pyskl/ckpt/posec3d/slowonly_r50_ntu120_xsub/joint.pth',
        help='skeleton action recognition checkpoint file/url')
    parser.add_argument(
        '--det-config',
        default='demo/faster_rcnn_r50_fpn_1x_coco-person.py',
        help='human detection config file path (from mmdet)')
    parser.add_argument(
        '--det-checkpoint',
        default=('https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/'
                 'faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth'),
        help='human detection checkpoint file/url')
    parser.add_argument(
        '--pose-config',
        default='demo/hrnet_w32_coco_256x192.py',
        help='human pose estimation config file path (from mmpose)')
    parser.add_argument(
        '--pose-checkpoint',
        default='https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth',
        help='human pose estimation checkpoint file/url')
    parser.add_argument(
        '--det-score-thr',
        type=float,
        default=0.9,
        help='the threshold of human detection score')
    parser.add_argument(
        '--label-map',
        default='tools/data/label_map/nturgbd_120.txt',
        help='label map file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--short-side',
        type=int,
        default=480,
        help='specify the short-side length of the image')
    args = parser.parse_args()
    return args

import os.path as osp
import imageio


from cnocr import CnOcr

ocr = CnOcr()  # 所有参数都使用默认值




def frame_extraction(video_path ,subsam=1,get =None):
    target_dir = osp.join('./tmp/', osp.basename(osp.splitext(video_path)[0]))

    os.makedirs(target_dir, exist_ok=True)
    frame_tmpl = osp.join(target_dir, 'img_{:06d}.jpg')

    # Load the video using imageio
    video = imageio.get_reader(video_path)
    fps = video.get_meta_data()['fps'] // subsam
    frames = []
    frame_paths = []
    cnt = 0
    prog_bar = mmcv.ProgressBar(video.count_frames())
    for idx, frame in enumerate(video):

        # print(frame.shape  )

        # out = ocr.ocr(frame[:200,:1000,::-1])
        # # out = ocr.ocr(frame)

        # print(out)
        # print(idx)
        if idx % subsam == 0:
            frame_path = frame_tmpl.format(cnt + 1)
            cv2.imwrite(frame_path, frame[:,:,::-1])
            frame_paths.append(frame_path)
            cnt += 1
            prog_bar.update()
        if get is not None and idx >= get:
            break
    return frame_paths ,fps


def main():
    args = parse_args()
    video_name = osp.basename( osp.splitext(args.video)[0])

    # folder = osp.dirname(args.video)
    args.out_filename = osp.join(osp.dirname(args.video), 'split')

    print(args.out_filename)
    frame_paths ,fps = frame_extraction(args.video,get =300)
    num_frame = len(frame_paths)
    print('frame len:{}  {}fps'.format(num_frame,fps))

    # config = mmcv.Config.fromfile(args.config)
    # config.data.test.pipeline = [x for x in config.data.test.pipeline if x['type'] != 'DecompressPose']

    torch.cuda.empty_cache()

    target_dir = osp.join(args.out_filename,video_name )
    os.makedirs(target_dir, exist_ok=True)
    print(target_dir)


    save_dir = osp.join(target_dir , video_name + '_{}_{}.mp4'.format(0,0))
    vid = mpy.ImageSequenceClip([ x for x in frame_paths ], fps=fps)
    print('vid shape :',vid.size)
    vid.write_videofile(save_dir , remove_temp=True)

    tmp_frame_dir = osp.dirname(frame_paths[0])
    shutil.rmtree(tmp_frame_dir)

if __name__ == '__main__':
    main()
