from re import A
import cv2, logging, os,argparse,sys,yaml
import torch
import numpy as np
 
ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
sys.path.append(ROOT)
from src import utils as U
from src.generator import Generator
from src.processor import Processor
from src.visualizer import Visualizer
import src.model as model
from src.dataset.graph import Graph






def init_parameters():
    parser = argparse.ArgumentParser(description='Skeleton-based Action Recognition')

    # Setting
    
    parser.add_argument('--device', '-g', type=str, nargs='+', default='cpu', help='Using GPUs') #Sử dụng GPU
    parser.add_argument('--seed', '-s', type=int, default=1, help='Random seed') # thoi gian nghi

    parser.add_argument('--type_test', type=int, default=0,help='0: video, 1: camera')
    parser.add_argument('--video', type=str, default='ResGCNv1/Demo/1a (online-video-cutter.com).mp4',help='path video test')

    parser.add_argument('--pretrained_path', '-pp', type=str, default='ResGCNv1/pretrained/1007_resgcn-b19_ntu-xsub120.pth.tar', help='Path to pretrained models')#Đường dẫn đến các mô hình được đào tạo trước
    parser.add_argument('--work_dir', '-w', type=str, default='', help='Work dir')
    
    parser.add_argument('--path_output', '-p', type=str, default='ResGCNv1/Demo/output', help='Path to save preprocessed skeleton files')#Đường dẫn để lưu các tệp bộ xương được xử lý trước
    parser.add_argument('--label_map',type=str,default='ResGCNv1/Demo/utils/label_map.txt',help='Path to label map file')
   
    
    parser.add_argument('--evaluate', '-e', default=False, action='store_true', help='Evaluate')  #Đánh giá
    parser.add_argument('--extract', '-ex', default=False, action='store_true', help='Extract')#Trích xuất
    
    # Dataloader
    parser.add_argument('--dataset', '-d', type=str, default='ntu-xsub120', help='Select dataset')#Chọn tập dữ liệu
    # parser.add_argument('--dataset_args', default=dict(), help='Args for creating dataset')#Args để tạo tập dữ liệu
    

    # Model
    parser.add_argument('--model_type', '-mt', type=str, default='resgcn-b19', help='Model type')
    parser.add_argument('--model_args', default=[9,2], help='Args for creating model')#Args để tạo tập dữ liệu
    #Config model
    parser.add_argument('--pose_config', type=str, default='ResGCNv1/Demo/utils/pose_config.py', help='Path to config file')
    parser.add_argument('--pose_checkpoint', default='ResGCNv1/Demo/utils/pose_checkpoint.pth', help='Args for config file')
    parser.add_argument('--det_config', type=str, default='ResGCNv1/Demo/utils/det_config.py', help='Path to config file')
    parser.add_argument('--det_checkpoint', default='ResGCNv1/Demo/utils/det_checkpoint.pth', help='Args for config file')
    parser.add_argument('--skeleton_config', type=str, default='ResGCNv1/Demo/utils/skeletion_config.py', help='Path to config file')
    parser.add_argument('--skeleton_checkpoint', default='ResGCNv1/Demo/utils/skeletion_checkpoint.pth', help='Args for config file')
    parser.add_argument('--rgb_stdet_config', default='', help='Args for config file')
    parser.add_argument('--rgb_stdet_checkpoint', default='', help='Args for config file')
    parser.add_argument('--cfg_options', default={}, help='Args for config file')
    parser.add_argument('--rgb_config', default='', help='Args for config file')
    parser.add_argument('--rgb_checkpoint', default='', help='Args for config file')
    parser.add_argument('--skeleton_stdet_checkpoint', default='ResGCNv1/Demo/utils/skeletion_stdet_checkpoint.pth', help='Args for config file')
    parser.add_argument('--threshold',type=float,default=0.01,help='recognition score threshold')
    parser.add_argument('--average-size',type=int,default=1,help='number of latest clips to be averaged for prediction')
    parser.add_argument('--drawing-fps',type=int,default=29, help='Set upper bound FPS value of the output drawing')
    parser.add_argument('--inference-fps',type=int,default=4, help='Set upper bound FPS value of model inference')
    return parser



def init_model(args):
    graph = Graph(args.dataset)
    
    kwargs = {
            'data_shape': [3,6,300,25,2],
            'num_class': 120,
            'A': torch.Tensor(graph.A),
            'parts': [torch.Tensor(part).long() for part in graph.parts]
    }
            
    
    output_device = None
    if output_device is not None:
        device =  torch.device('cuda:{}'.format(output_device))
    else:
        device =  torch.device('cpu')

    net=model.create(args.model_type,**kwargs).to(device)
    net=torch.nn.DataParallel(net,'cpu',output_device=output_device)
    
    logging.info('Model: {} {}'.format(args.model_type, args.model_args))
    logging.info('Model parameters: {:.2f}M'.format(
        sum(p.numel() for p in net.parameters()) / 1000 / 1000
    ))
    
    pretrained_model=args.pretrained_path
    if os.path.exists(pretrained_model):
        checkpoint = torch.load(pretrained_model, map_location=torch.device('cpu'))
        net=net.module.load_state_dict(checkpoint['model'])
        logging.info('Pretrained model: {}'.format(pretrained_model))
    elif args.pretrained_path:
        logging.warning('Warning: Do NOT exist this pretrained model: {}'.format(pretrained_model))
    
    return net

def load_label_map(file_path):
    """Load Label Map.
    Args:
        file_path (str): The file path of label map.
    Returns:
        dict: The label map (int -> label name).
    """
    lines = open(file_path).readlines()
    lines = [x.strip().split(': ') for x in lines]
    return {int(x[0]): x[1] for x in lines}

def hex2color(h):
    """Convert the 6-digit hex string to tuple of 3 int value (RGB)"""
    return (int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16))


def expand_bbox(bbox, h, w, ratio=1.25):
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    width = x2 - x1
    height = y2 - y1

    square_l = max(width, height)
    new_width = new_height = square_l * ratio

    new_x1 = max(0, int(center_x - new_width / 2))
    new_x2 = min(int(center_x + new_width / 2), w)
    new_y1 = max(0, int(center_y - new_height / 2))
    new_y2 = min(int(center_y + new_height / 2), h)
    return (new_x1, new_y1, new_x2, new_y2)


def cal_iou(box1, box2):
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2

    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    intersect = w * h
    union = s1 + s2 - intersect
    iou = intersect / union

    return iou





def read_yaml(filepath):
    ''' Input a string filepath, 
        output a `dict` containing the contents of the yaml file.
    '''
    with open(filepath, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    return data_loaded

def abbrev(name):
    """Get the abbreviation of label name:
    'take (an object) from (a person)' -> 'take ... from ...'
    """
    while name.find('(') != -1:
        st, ed = name.find('('), name.find(')')
        name = name[:st] + '...' + name[ed + 1:]
    return name

def dense_timestamps(timestamps, n):
    """Make it nx frames."""
    old_frame_interval = (timestamps[1] - timestamps[0])
    start = timestamps[0] - old_frame_interval / n * (n - 1) / 2
    new_frame_inds = np.arange(
        len(timestamps) * n) * old_frame_interval / n + start
    return new_frame_inds.astype(np.int)

    
def pack_results(human_detection,result,img_h,img_w):
    human_detection[:, 0::2] /= img_w
    human_detection[:, 1::2] /= img_h
    results = []
    if result is None:
        return None
    for prop, res in zip(human_detection, result):
        res.sort(key=lambda x: -x[1])
        results.append(
            (prop.data.cpu().numpy(), [x[0] for x in res], [x[1]
                                                            for x in res]))
    return results