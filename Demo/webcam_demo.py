import argparse
from cmath import pi
import time
from collections import deque
from operator import itemgetter
from threading import Thread

import cv2
import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.parallel import collate, scatter

from mmaction.apis import init_recognizer
from mmaction.datasets.pipelines import Compose

FONTFACE = cv2.FONT_HERSHEY_COMPLEX_SMALL
FONTSCALE = 1
FONTCOLOR = (255, 255, 255)  # BGR, white
MSGCOLOR = (128, 128, 128)  # BGR, gray
THICKNESS = 1
LINETYPE = 1

EXCLUED_STEPS = [
    'OpenCVInit', 'OpenCVDecode', 'DecordInit', 'DecordDecode', 'PyAVInit',
    'PyAVDecode', 'RawFrameDecode'
]

# from clearml import Task,Logger
# from tensorboardX import SummaryWriter
from Utils import init_parameters,init_model,\
    read_yaml,hex2color,cal_iou,pack_results,abbrev,expand_bbox,load_label_map,dense_timestamps









def show_results(args):
    print('Press "Esc", "q" or "Q" to exit')
    text_info={}
    cur_time=time.time()
    while True:
        msg='Waiting for action...'
        _,frame=camera.read()
        frame_queue.append(np.array(frame[:,:,::-1]))

        if len(result_queue)!=0:
            text_info={}
            result_queue=result_queue.popleft()
            for i,result in enumerate(results):
                select_label,score=result
                if score< args.threshold:
                    break
                location=(0,40+i*20)
                text=select_label+ ": "+ str(round(score,2))
                text_info[location]=text
                cv2.putText(frame,text,location,FONTFACE,FONTSCALE.FONTCOLOR,THICKNESS, LINETYPE)
        elif len(text_info)!=0:
            for location, text in text_info.items():
                cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                            FONTCOLOR, THICKNESS, LINETYPE)
        else:
            cv2.putText(frame, msg, (0, 40), FONTFACE, FONTSCALE, MSGCOLOR,
                        THICKNESS, LINETYPE)
        
        cv2.imshow('camera',frame)
        ch=cv2.waitKey(1)

        if ch==27 or ch == ord('q') or ch == ord('Q'):
                break
        if args.drawing_fps>0:
            sleep_time=1/args.drawing_fps - (time.time()-cur_time)
            if sleep_time>0:
                time.sleep(sleep_time)
            cur_time=time.time()

def inference(args):
    score_cache=deque()
    scores_sum=0
    cur_time= time.time()
    while True:
        cur_windows=[]
        while len(cur_windows)==0:
            if len(frame_queue)==sample_length:
                cur_windows=list(np.array(frame_queue))
                if data['img_shape'] is None:
                    data['img_shape']= frame_queue.popleft().shape[:2]
    
        cur_data=data.copy()
        cur_data['imgs']=cur_windows
        cur_data=test_pipeline(cur_data)
        cur_data= collate([cur_data],samples_per_gpu=1)
        if next(model.parameters()).is_cuda:
            cur_data=scatter(cur_data,[device])[0]
        with torch.no_grad():
            scores = model(return_loss=False, **cur_data)[0]

        score_cache.append(scores)
        scores_sum += scores

        if len(score_cache) == args.average_size:
            scores_avg = scores_sum / args.average_size
            num_selected_labels = min(len(label), 5)

            scores_tuples = tuple(zip(label, scores_avg))
            scores_sorted = sorted(
                scores_tuples, key=itemgetter(1), reverse=True)
            results = scores_sorted[:num_selected_labels]

            result_queue.append(results)
            scores_sum -= score_cache.popleft()

        if args.inference_fps > 0:
            # add a limiter for actual inference fps <= args.inference_fps
            sleep_time = 1 / args.inference_fps - (time.time() - cur_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            cur_time = time.time()

    camera.release()
    cv2.destroyAllWindows()

def main():
    parser = init_parameters()
    args, _ = parser.parse_known_args()
    global frame_queue, camera, frame, results,  sample_length, \
        data, test_pipeline, model,  label, device,result_queue
    
    device = torch.device(args.device)
    cfg = Config.fromfile(args.skeleton_config)
    cfg.merge_from_dict(args.cfg_options)

    model = init_recognizer(cfg, args.skeleton_checkpoint, device=device)
    camera = cv2.VideoCapture(args.video)
    data = dict(img_shape=None, modality='RGB', label=-1)

    with open(args.label_map, 'r') as f:
        label = [line.strip() for line in f]

    cfg= model.cfg
    sample_length=0
    pipeline=cfg.data.test.pipeline
    pipeline_=pipeline.copy()
    for step in pipeline:
        if 'SampleFrames' in step['type']:
            sample_length=step['clip_len']* step['num_clips']
            data['num_clips']= step['num_clips']
            data['clip_len']= step['num_clips']
            pipeline_.remove(step)
        if step['type'] in EXCLUED_STEPS:
            pipeline_.remove(step)
    test_pipeline = Compose(pipeline_)

    assert sample_length > 0

    try:
        frame_queue = deque(maxlen=sample_length)
        result_queue = deque(maxlen=1)
        pw = Thread(target=show_results(args), args=(), daemon=True)
        pr = Thread(target=inference(args), args=(), daemon=True)
        pw.start()
        pr.start()
        pw.join()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
