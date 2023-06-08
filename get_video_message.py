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

def parse_args():
    parser = argparse.ArgumentParser(description='PoseC3D demo')
    # parser.add_argument('--video', default='split/hiv00047.mp4', help='video file/url')
    parser.add_argument('--path', default='data/64/hiv00001.mp4', help='video file/url')
    # parser.add_argument('--out_filename', default='split/out/', help='output filename')
    # parser.add_argument(
    #     '--config',
    #     default='configs/posec3d/slowonly_r50_ntu120_xsub/joint.py',
    #     help='skeleton action recognition config file path')
    # parser.add_argument(
    #     '--checkpoint',
    #     default='https://download.openmmlab.com/mmaction/pyskl/ckpt/posec3d/slowonly_r50_ntu120_xsub/joint.pth',
    #     help='skeleton action recognition checkpoint file/url')
    # parser.add_argument(
    #     '--det-config',
    #     default='demo/faster_rcnn_r50_fpn_1x_coco-person.py',
    #     help='human detection config file path (from mmdet)')
    # parser.add_argument(
    #     '--det-checkpoint',
    #     default=('https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/'
    #              'faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth'),
    #     help='human detection checkpoint file/url')
    # parser.add_argument(
    #     '--pose-config',
    #     default='demo/hrnet_w32_coco_256x192.py',
    #     help='human pose estimation config file path (from mmpose)')
    # parser.add_argument(
    #     '--pose-checkpoint',
    #     default='https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth',
    #     help='human pose estimation checkpoint file/url')
    # parser.add_argument(
    #     '--det-score-thr',
    #     type=float,
    #     default=0.9,
    #     help='the threshold of human detection score')
    # parser.add_argument(
    #     '--label-map',
    #     default='tools/data/label_map/nturgbd_120.txt',
    #     help='label map file')
    # parser.add_argument(
    #     '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    # parser.add_argument(
    #     '--short-side',
    #     type=int,
    #     default=480,
    #     help='specify the short-side length of the image')
    args = parser.parse_args()
    return args

import os.path as osp
import imageio
from cnocr import CnOcr
import re

ocr = CnOcr(rec_model_name='ch_PP-OCRv3')  # 所有参数都使用默认值




def get_one_time(frame):
    out = ocr.ocr(frame[:200,:1000,::])
    print(out)
    
    out = out[0]
    txt =out['text'].replace(" ", "").replace('：','')
    txt = re.sub(r"[\u4e00-\u9fa5]+", "", txt)

    try:
        if len(txt) == 14:
            txt = eval(txt)
            # print('time:',txt)
            return txt
    except:
        pass
        # print('error:',txt) 
        return None

def video_msg(video_path ):

    os.makedirs('./tmp1', exist_ok=True)
    frame_tmpl = osp.join('./tmp1', 'img_{:06d}.jpg')

    # Load the video using imageio
    print(video_path)
    video = imageio.get_reader(video_path)
    fps = video.get_meta_data()['fps'] 
    print(video.get_meta_data().keys())
    for i in video.get_meta_data().keys():
        print('{}--{}'.format(i, video.get_meta_data()[i]))

    print('frame count:',video.count_frames()-1)
    # start_time =None

    for idx, frame in enumerate(video):
                # print(idx)
        frame_path = frame_tmpl.format(idx + 1)
        cv2.imwrite(frame_path, frame)
       
        # if start_time is None:
        #     start_time = get_one_time(frame )

        #     if start_time is not None:
        #         break



    # 打开视频文件
    video = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not video.isOpened():
        print("无法打开视频文件")
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_index  = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) 

    print(fps,width,height,frame_index)

    success, frame = video.read()

    cnt =0
    while success:
        # print(idx)
        frame_path = frame_tmpl.format(cnt + 1)
        cv2.imwrite(frame_path, frame)
        cnt += 1
        success, frame = video.read()


    print(frame.shape)
    start_time = get_one_time(frame )

    # 设置要读取的帧索引（例如，读取第100帧）
    

    # 读取指定帧的图像
    # video.set(cv2.CAP_PROP_POS_FRAMES, 100)
    # success, frame = video.read()
    # if success:
    #     print(frame.shape)
    #     endtime = get_one_time(frame )


    video.release()


    endtime = get_one_time(frame )
    if endtime is not None:
        print('{}-{}'.format(start_time,endtime)  )




    return fps


def main():
    args = parse_args()
    fps = video_msg(args.path)

if __name__ == '__main__':
    main()
