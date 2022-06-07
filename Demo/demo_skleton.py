
import copy as cp
import os,cv2,sys
from unittest import result
import os.path as osp
from matplotlib.pyplot import flag
import numpy as np
import cv2
from torch import diag
from inference import Inference_detector,Init_detector

import logging
import torch
import threading
import shutil
import mmcv
from mmcv import DictAction
from mmcv.runner import load_checkpoint

from mmaction.apis import inference_recognizer
from mmaction.datasets.pipelines import Compose
from mmaction.models import build_detector, build_model, build_recognizer


try:
    from mmdet.apis import inference_detector, init_detector
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_detector` and '
                      '`init_detector` form `mmdet.apis`. These apis are '
                      'required in this demo! ')

try:
    from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                             vis_pose_result)
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_top_down_pose_model`, '
                      '`init_pose_model`, and `vis_pose_result` form '
                      '`mmpose.apis`. These apis are required in this demo! ')

try:
    import moviepy.editor as mpy
except ImportError:
    raise ImportError('Please install moviepy to enable output file')

# from clearml import Task,Logger
# from tensorboardX import SummaryWriter
from Utils import init_parameters,init_model,\
    read_yaml,hex2color,cal_iou,pack_results,abbrev,expand_bbox,load_label_map,dense_timestamps



FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.5
FONTCOLOR = (255, 255, 255)  # BGR, white
MSGCOLOR = (128, 128, 128)  # BGR, gray
THICKNESS = 1
LINETYPE = 1

PLATEBLUE = '03045e-023e8a-0077b6-0096c7-00b4d8-48cae4'
PLATEBLUE = PLATEBLUE.split('-')
PLATEBLUE = [hex2color(h) for h in PLATEBLUE]
PLATEGREEN = '004b23-006400-007200-008000-38b000-70e000'
PLATEGREEN = PLATEGREEN.split('-')
PLATEGREEN = [hex2color(h) for h in PLATEGREEN]



# task = Task.init(project_name='Action Recognition', task_name='task_1')
# S_writer=SummaryWriter('run/Action Recognition')


def visualize(frames,
            annotations,
            pose_results,
            action_results,
            pose_model,
            plate=PLATEBLUE,
            max_num=5):
    """
    Visualize frames with predicted annotations.
    Args:
        frames (list[np.ndarray]): Frames for visualization, note that
            len(frames) % len(annotations) should be 0.
        annotations (list[list[tuple]]): The predicted spatio-temporal
            detection results.
        pose_results (list[list[tuple]): The pose results.
        action_result (str): The predicted action recognition results.
        pose_model (nn.Module): The constructed pose model.
        plate (str): The plate used for visualization. Default: PLATEBLUE.
        max_num (int): Max number of labels to visualize for a person box.
            Default: 5.
    Returns:
        list[np.ndarray]: Visualized frames.
    """
    assert max_num +1 <= len(plate)
    plate=[x[::1] for x in plate]
    frames_=cp.deepcopy(frames)
    nf,na=len(frames),len(annotations)
    assert nf%na==0
    nfpa=len(frames)//len(annotations)
    anno=None
    h,w,_=frames[0].shape
    scale_ration=np.array([w,h,w,h])

    if pose_results:
        for i in range(nf):
            frames_[i]=vis_pose_result(pose_model,frames_[i],pose_results[i])
    
    for i in range(na):
        anno= annotations[i]
        if anno is None:
            continue
        for j in range(len(nfpa)):
            ind=i*nfpa+j
            frame=frames_[ind]

            cv2.putText(frame,action_results,(10,30),FONTFACE,FONTSCALE,FONTCOLOR,THICKNESS,LINETYPE)
            for ann in anno:
                box=ann[0]
                label=ann[1]
                if not len(label):
                    continue
                score=ann[2]
                box=(box*scale_ration).astype(np.int64)
                st,ed=tuple(box[:2]),tuple(box[2:])
                if not pose_results:
                    cv2.rectangle(frame,st,ed,plate[0],2)
                for k,lb in enumerate(label):
                    if k>=max_num:
                        break
                    text=abbrev(lb)
                    text=': '.join([text,str(score[k])])
                    location=(0 +st[0],18+k*18+st[1])
                    textsize=cv2.getTextSize(text,FONTFACE,FONTSCALE,THICKNESS)[0]
                    textwidth=textsize[0]
                    diag0=(location[0]+textwidth,location[1]-14)
                    diag1=(location[0],location[1]+2)
                    cv2.rectangle(frame,location,diag0,plate[k+1],-1)
                    cv2.putText(frame,text,location,FONTFACE,FONTSCALE,FONTCOLOR,THICKNESS,LINETYPE)
    return frames_





def frame_extraction(video_path,short_side=480):
    """
    Execute the frames of the video
    Args: video_path: path of the video
    """
    target_dir=osp.join('ResGCNv1/tmp', osp.basename(osp.splitext(video_path)[0]))
    os.makedirs(target_dir, exist_ok=True)

    frame_tmpl=osp.join(target_dir,'{:06d}.jpg')
    vid=cv2.VideoCapture(video_path)

    frames=[]
    frame_paths=[]
    flag,frame=vid.read()
    cnt=0
    

    while flag:
        frames.append(frame)
        frame_path=frame_tmpl.format(cnt+1)
        frame_paths.append(frame_path)
        cv2.imwrite(frame_path,frame)
        cnt+=1
        flag,frame=vid.read()

    return frame_paths, frames

def detection_inference(args, frame_paths):
    """Detect human boxes given frame paths.
    Args:
        args (argparse.Namespace): The arguments.
        frame_paths (list[str]): The paths of frames to do detection inference.
    Returns:
        list[np.ndarray]: The human detection results.
    """
    model = init_detector(args.det_config, args.det_checkpoint, args.device)
    
    results = []
    print('Performing Human Detection for each frame')
    prog_bar = mmcv.ProgressBar(len(frame_paths))
    for frame_path in frame_paths:
        result = inference_detector(model, frame_path)
        # We only keep human detections with score larger than det_score_thr
        result = result[0][result[0][:, 4] >= 0.9]
        results.append(result)
        prog_bar.update()

    return results


def pose_inference(args,pose_model, frame_paths, det_results):
    
    ret = []
    print('Performing Human Pose Estimation for each frame')
    prog_bar = mmcv.ProgressBar(len(frame_paths))
    for f, d in zip(frame_paths, det_results):
        # Align input format
        d = [dict(bbox=x) for x in list(d)]

        pose = inference_top_down_pose_model(pose_model, f, d, format='xyxy')[0]
        ret.append(pose)
        prog_bar.update()
    return ret    

def skeleton_based_action_recognition(args, pose_results, num_frame, h, w):
    fake_anno = dict(
        frame_dict='',
        label=-1,
        img_shape=(h, w),
        origin_shape=(h, w),
        start_index=0,
        modality='Pose',
        total_frames=num_frame)
    num_person = max([len(x) for x in pose_results])

    num_keypoint = 17
    keypoint = np.zeros((num_person, num_frame, num_keypoint, 2),
                        dtype=np.float16)
    keypoint_score = np.zeros((num_person, num_frame, num_keypoint),
                              dtype=np.float16)
    for i, poses in enumerate(pose_results):
        for j, pose in enumerate(poses):
            pose = pose['keypoints']
            keypoint[j, i] = pose[:, :2]
            keypoint_score[j, i] = pose[:, 2]

    fake_anno['keypoint'] = keypoint
    fake_anno['keypoint_score'] = keypoint_score

    label_map = [x.strip() for x in open(args.label_map).readlines()]
    num_class = len(label_map)

    skeleton_config = mmcv.Config.fromfile(args.skeleton_config)
    skeleton_config.model.cls_head.num_classes = num_class  # for K400 dataset
    skeleton_pipeline = Compose(skeleton_config.test_pipeline)
    skeleton_imgs = skeleton_pipeline(fake_anno)['imgs'][None]
    skeleton_imgs = skeleton_imgs.to(args.device)

    # Build skeleton-based recognition model
    skeleton_model = build_model(skeleton_config.model)
    load_checkpoint(
        skeleton_model, args.skeleton_checkpoint, map_location='cpu')
    skeleton_model.to(args.device)
    skeleton_model.eval()

    with torch.no_grad():
        output = skeleton_model(return_loss=False, imgs=skeleton_imgs)

    action_idx = np.argmax(output)
    skeleton_action_result = label_map[action_idx]  # skeleton-based action result for the whole video

    return skeleton_action_result



def rgb_based_action_recognition(args):
    rgb_config = mmcv.Config.fromfile(args.rgb_config)
    rgb_config.model.backbone.pretrained = None
    rgb_model = build_recognizer(
        rgb_config.model, test_cfg=rgb_config.get('test_cfg'))
    load_checkpoint(rgb_model, args.rgb_checkpoint, map_location='cpu')
    rgb_model.cfg = rgb_config
    rgb_model.to(args.device)
    rgb_model.eval()
    action_results = inference_recognizer(
        rgb_model, args.video, label_path=args.label_map)
    rgb_action_result = action_results[0][0]
    return rgb_action_result


def skeleton_based_stdet(args, label_map, human_detections, pose_results,
                         num_frame, clip_len, frame_interval, h, w):
    window_size = clip_len * frame_interval
    assert clip_len % 2 == 0, 'We would like to have an even clip_len'
    timestamps = np.arange(window_size // 2, num_frame + 1 - window_size // 2,
                           args.predict_stepsize)

    skeleton_config = mmcv.Config.fromfile(args.skeleton_config)
    num_class = max(label_map.keys()) + 1  # for AVA dataset (81)
    skeleton_config.model.cls_head.num_classes = num_class
    skeleton_pipeline = Compose(skeleton_config.test_pipeline)
    skeleton_stdet_model = build_model(skeleton_config.model)
    load_checkpoint(
        skeleton_stdet_model,
        args.skeleton_stdet_checkpoint,
        map_location='cpu')
    skeleton_stdet_model.to(args.device)
    skeleton_stdet_model.eval()

    skeleton_predictions = []

    print('Performing SpatioTemporal Action Detection for each clip')
    prog_bar = mmcv.ProgressBar(len(timestamps))
    for timestamp in timestamps:
        proposal = human_detections[timestamp - 1]
        if proposal.shape[0] == 0:  # no people detected
            skeleton_predictions.append(None)
            continue

        start_frame = timestamp - (clip_len // 2 - 1) * frame_interval
        frame_inds = start_frame + np.arange(0, window_size, frame_interval)
        frame_inds = list(frame_inds - 1)
        num_frame = len(frame_inds)  # 30

        pose_result = [pose_results[ind] for ind in frame_inds]

        skeleton_prediction = []
        for i in range(proposal.shape[0]):  # num_person
            skeleton_prediction.append([])

            fake_anno = dict(
                frame_dict='',
                label=-1,
                img_shape=(h, w),
                origin_shape=(h, w),
                start_index=0,
                modality='Pose',
                total_frames=num_frame)
            num_person = 1

            num_keypoint = 17
            keypoint = np.zeros(
                (num_person, num_frame, num_keypoint, 2))  # M T V 2
            keypoint_score = np.zeros(
                (num_person, num_frame, num_keypoint))  # M T V

            # pose matching
            person_bbox = proposal[i][:4]
            area = expand_bbox(person_bbox, h, w)

            for j, poses in enumerate(pose_result):  # num_frame
                max_iou = float('-inf')
                index = -1
                if len(poses) == 0:
                    continue
                for k, per_pose in enumerate(poses):
                    iou = cal_iou(per_pose['bbox'][:4], area)
                    if max_iou < iou:
                        index = k
                        max_iou = iou
                keypoint[0, j] = poses[index]['keypoints'][:, :2]
                keypoint_score[0, j] = poses[index]['keypoints'][:, 2]

            fake_anno['keypoint'] = keypoint
            fake_anno['keypoint_score'] = keypoint_score

            skeleton_imgs = skeleton_pipeline(fake_anno)['imgs'][None]
            skeleton_imgs = skeleton_imgs.to(args.device)

            with torch.no_grad():
                output = skeleton_stdet_model(
                    return_loss=False, imgs=skeleton_imgs)
                output = output[0]
                for k in range(len(output)):  # 81
                    if k not in label_map:
                        continue
                    if output[k] > 0.4:
                        skeleton_prediction[i].append(
                            (label_map[k], output[k]))

        skeleton_predictions.append(skeleton_prediction)
        prog_bar.update()

    return timestamps, skeleton_predictions


# def rgb_based_stdet(args, frames, label_map, human_detections, w, h, new_w,
#                     new_h, w_ratio, h_ratio):

#     rgb_stdet_config = mmcv.Config.fromfile(args.rgb_stdet_config)
#     rgb_stdet_config.merge_from_dict(args.cfg_options)

#     val_pipeline = rgb_stdet_config.data.val.pipeline
#     sampler = [x for x in val_pipeline if x['type'] == 'SampleAVAFrames'][0]
#     clip_len, frame_interval = sampler['clip_len'], sampler['frame_interval']
#     assert clip_len % 2 == 0, 'We would like to have an even clip_len'

#     window_size = clip_len * frame_interval
#     num_frame = len(frames)
#     timestamps = np.arange(window_size // 2, num_frame + 1 - window_size // 2,
#                            args.predict_stepsize)

#     # Get img_norm_cfg
#     img_norm_cfg = rgb_stdet_config['img_norm_cfg']
#     if 'to_rgb' not in img_norm_cfg and 'to_bgr' in img_norm_cfg:
#         to_bgr = img_norm_cfg.pop('to_bgr')
#         img_norm_cfg['to_rgb'] = to_bgr
#     img_norm_cfg['mean'] = np.array(img_norm_cfg['mean'])
#     img_norm_cfg['std'] = np.array(img_norm_cfg['std'])

#     # Build STDET model
#     try:
#         # In our spatiotemporal detection demo, different actions should have
#         # the same number of bboxes.
#         rgb_stdet_config['model']['test_cfg']['rcnn']['action_thr'] = .0
#     except KeyError:
#         pass

#     rgb_stdet_config.model.backbone.pretrained = None
#     rgb_stdet_model = build_detector(
#         rgb_stdet_config.model, test_cfg=rgb_stdet_config.get('test_cfg'))

#     load_checkpoint(
#         rgb_stdet_model, args.rgb_stdet_checkpoint, map_location='cpu')
#     rgb_stdet_model.to(args.device)
#     rgb_stdet_model.eval()

#     predictions = []

#     print('Performing SpatioTemporal Action Detection for each clip')
#     prog_bar = mmcv.ProgressBar(len(timestamps))
#     for timestamp in timestamps:
#         proposal = human_detections[timestamp - 1]

#         if proposal.shape[0] == 0:
#             predictions.append(None)
#             continue

#         start_frame = timestamp - (clip_len // 2 - 1) * frame_interval
#         frame_inds = start_frame + np.arange(0, window_size, frame_interval)
#         frame_inds = list(frame_inds - 1)

#         imgs = [frames[ind].astype(np.float32) for ind in frame_inds]
#         _ = [mmcv.imnormalize_(img, **img_norm_cfg) for img in imgs]

#         input_array = np.stack(imgs).transpose((3, 0, 1, 2))[np.newaxis]
#         input_tensor = torch.from_numpy(input_array).to(args.device)

#         with torch.no_grad():
#             result = rgb_stdet_model(
#                 return_loss=False,
#                 img=[input_tensor],
#                 img_metas=[[dict(img_shape=(new_h, new_w))]],
#                 proposals=[[proposal]])
#             result = result[0]
#             prediction = []
#             # N proposals
#             for i in range(proposal.shape[0]):
#                 prediction.append([])

#             # Perform action score thr
#             for i in range(len(result)):  # 80
#                 if i + 1 not in label_map:
#                     continue
#                 for j in range(proposal.shape[0]):
#                     if result[i][j, 4] > 0.4:
#                         prediction[j].append((label_map[i + 1], result[i][j,4]))
#             predictions.append(prediction)
#         prog_bar.update()

#     return timestamps, predictions







def main():
    
    parser = init_parameters()
    args, _ = parser.parse_known_args()


    net=init_model(args)
    # print(net)
    
    
    
    print("====================Pose Estimation Done=================")
    frame_paths,original_frames=frame_extraction(args.video,short_side=480)
    num_frames = len(frame_paths)

    h,w,_= original_frames[0].shape

    pose_model = init_pose_model(args.pose_config, args.pose_checkpoint,args.device)
    
    human_detections= detection_inference(args, frame_paths)
    pose_results=None
    

    pose_results = pose_inference(args,pose_model, frame_paths, human_detections)
    

    
    new_w,new_h=mmcv.rescale_size((w,h), (256, np.Inf))
    frames=[mmcv.imresize(img,(new_w,new_h)) for img in original_frames]
    w_ratio, h_ratio=new_w/w,new_h/h

    # Load spatio-temporal detection label_map
    stdet_label_map = load_label_map(args.label_map_stdet)
    rgb_stdet_config = mmcv.Config.fromfile(args.rgb_stdet_config)
    rgb_stdet_config.merge_from_dict(args.cfg_options)

    
    print('Use skeleton-based recognition')
    action_result = skeleton_based_action_recognition(args, pose_results, num_frames, h, w)

    stdet_preds = None
    
    print('Use skeleton-based SpatioTemporal Action Detection')
    clip_len, frame_interval = 30, 1
    timestamps, stdet_preds = skeleton_based_stdet(args, stdet_label_map,
                                                    human_detections,
                                                    pose_results, num_frames,
                                                    clip_len,
                                                    frame_interval, h, w)
    for i in range(len(human_detections)):
        det = human_detections[i]
        det[:, 0:4:2] *= w_ratio
        det[:, 1:4:2] *= h_ratio
        human_detections[i] = torch.from_numpy(det[:, :4]).to(args.device)

    

    stdet_results = []
    for timestamp, prediction in zip(timestamps, stdet_preds):
        human_detection = human_detections[timestamp - 1]
        stdet_results.append(
            pack_results(human_detection, prediction, new_h, new_w))

    
    predict_stepsize=8
    output_stepsize=1
    dense_n = int(predict_stepsize / output_stepsize)
    output_timestamps = dense_timestamps(timestamps, dense_n)
    frames = [
        cv2.imread(frame_paths[timestamp - 1])
        for timestamp in output_timestamps
    ]

    print('Performing visualization')
    # pose_model = init_pose_model(args.pose_config, args.pose_checkpoint,
    #                              args.device)

    
    pose_results = [pose_results[timestamp - 1] for timestamp in output_timestamps]

    vis_frames = visualize(frames, stdet_results, pose_results, action_result,
                           pose_model)
    vid = mpy.ImageSequenceClip([x[:, :, ::-1] for x in vis_frames],
                                fps=30)
    vid.write_videofile(args.path_output)

    tmp_frame_dir = osp.dirname(frame_paths[0])
    shutil.rmtree(tmp_frame_dir)
    
    
    print("====================Recognition Done=================")

if __name__ == '__main__':
    main()
    