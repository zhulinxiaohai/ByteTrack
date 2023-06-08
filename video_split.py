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
    parser.add_argument('--video', default='data/64/hiv00002.mp4', help='video file/url')
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

def frame_extraction(video_path ,subsam=1):
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
        # print(idx)
        if idx % subsam == 0:
            frame_path = frame_tmpl.format(cnt + 1)
            cv2.imwrite(frame_path, frame[:,:,::-1])
            frame_paths.append(frame_path)
            cnt += 1
            prog_bar.update()
    return frame_paths ,fps

def detection_inference(args, frames):
    """Detect human boxes given frame paths.
    Args:
        args (argparse.Namespace): The arguments.
        frame_paths (list[str]): The paths of frames to do detection inference.

    Returns:
        list[np.ndarray]: The human detection results.
    """
    model = init_detector(args.det_config, args.det_checkpoint, args.device)
    assert model is not None, ('Failed to build the detection model. Check if you have installed mmcv-full properly. '
                               'You should first install mmcv-full successfully, then install mmdet, mmpose. ')
    assert model.CLASSES[0] == 'person', 'We require you to use a detector trained on COCO'
    results = []
    print('Performing Human Detection for each frame')
    prog_bar = mmcv.ProgressBar(len(frames))
    index = 0 
    person =[]
    for frame in frames:
        index += 1
        result = inference_detector(model,  frame )
        # print('result:',result)
        person_sign = 0
        for i in result[0]:
            if i[4] >= args.det_score_thr:
                person_sign = 1
                break
        person.append(person_sign)

        # # We only keep human detections with score larger than det_score_thr
        result = result[0] [result[0][:, 4] >= args.det_score_thr]
        results.append(result)
        prog_bar.update()
    return person ,results


def main():
    args = parse_args()
    video_name = osp.basename( osp.splitext(args.video)[0])

    # folder = osp.dirname(args.video)
    args.out_filename = osp.join(osp.dirname(args.video), 'split')

    print(args.out_filename)
    frame_paths ,fps = frame_extraction(args.video)
    num_frame = len(frame_paths)
    print('frame len:{}  {}fps'.format(num_frame,fps))

    config = mmcv.Config.fromfile(args.config)
    config.data.test.pipeline = [x for x in config.data.test.pipeline if x['type'] != 'DecompressPose']

    # # Get Human detection results
    isperson ,det_results = detection_inference(args, frame_paths)
    print('isperson:',len(isperson) )

    start = 0
    end =0
    zero = 0
    out_frames = {0:[],1:[]}
    
    for i in range(len(isperson)):
        if isperson[i] == isperson[start]:
            end = i+1
            zero =0
        else:  # 1-->0
            zero += 1
            if zero >= 30:  #需要剪辑重新开始
                zero = 0
                if end - start > 90: #大于90帧才保存
                    out_frames[isperson[start]] .append([start ,end])
                    start =end
                    
                else:  #小于90帧,丢掉
                    start =end
    if end - start > 90: #大于50帧才保存
        out_frames[isperson[start]] .append([start ,end])                
    print(out_frames)
    torch.cuda.empty_cache()

    target_dir = osp.join(args.out_filename,video_name )
    os.makedirs(target_dir, exist_ok=True)
    print(target_dir)

    for pos_neg in [0,1]:
        for i  in range(len(out_frames[pos_neg])):
            index = out_frames[pos_neg][i]
            save_dir = osp.join(target_dir , video_name + '_{}_{}.mp4'.format(pos_neg,i))
            vid = mpy.ImageSequenceClip([ x for x in frame_paths[index[0]:index[1]] ], fps=fps)
            print('vid shape :',vid.size)
            vid.write_videofile(save_dir , remove_temp=True)

    tmp_frame_dir = osp.dirname(frame_paths[0])
    shutil.rmtree(tmp_frame_dir)

if __name__ == '__main__':
    main()
