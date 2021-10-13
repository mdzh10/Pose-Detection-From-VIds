# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from argparse import ArgumentParser
import numpy as np
import concurrent.futures
import cv2

from mmpose.apis import (get_track_id, inference_top_down_pose_model,
                         init_pose_model, process_mmdet_results,
                         vis_pose_tracking_result)
from mmpose.datasets import DatasetInfo

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

# modifcations: 
# helper functions 

def vid_fps_duration_frames(filename):
    video = cv2.VideoCapture(filename)

    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count/fps

    return duration, fps, frame_count

# extracts a specific frame from a video
def extract_frames(vid, frame_no):
    
    cap= cv2.VideoCapture(vid)
    len1 = vid_fps_duration_frames(vid)
    len1 = int(len1[2])

    if (frame_no < len1):
        cap.set(1, frame_no); # Where frame_no is the frame you want
        ret, frame = cap.read() # Read the frame

        cap.release()
        cv2.destroyAllWindows()
    
        return frame
        
    else:
        # handling blank frames
        frame = np.zeros((1920,1080,3), np.uint8)
        cap.release()
        cv2.destroyAllWindows()
        
        return frame

# defining combining frames function with multi threading 
def combining_frames(vids, image_size, frame_no):
  # extracting frames using 4 threads
    with concurrent.futures.ThreadPoolExecutor(max_workers = 4) as executor:
        futures = [ executor.submit(extract_frames, vids[j], frame_no) for j in range(4) ]
        return_value = [ future.result() for future in futures ] 
          
    f1 = np.array(return_value[0])
    f2 = np.array(return_value[1])
    f3 = np.array(return_value[2])
    f4 = np.array(return_value[3])

    f1 = cv2.resize(f1, image_size, interpolation = cv2.INTER_AREA)
    f2 = cv2.resize(f2, image_size, interpolation = cv2.INTER_AREA)
    f3 = cv2.resize(f3, image_size, interpolation = cv2.INTER_AREA)
    f4 = cv2.resize(f4, image_size, interpolation = cv2.INTER_AREA)
      
    final1 = cv2.hconcat([f1, f2])
    final2 = cv2.hconcat([f3, f4])
    final3 = cv2.vconcat([final1, final2])
    final_resized = cv2.resize(final3, image_size, interpolation = cv2.INTER_AREA)

    return final_resized


def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--video-path', type=str, help='Video path')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')
    parser.add_argument(
        '--out-video-root',
        default='',
        help='Root of the output video file. '
        'Default not saving the visualization video.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--use-oks-tracking', action='store_true', help='Using OKS tracking')
    parser.add_argument(
        '--tracking-thr', type=float, default=0.3, help='Tracking threshold')
    parser.add_argument(
        '--euro',
        action='store_true',
        help='Using One_Euro_Filter for smoothing')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    assert args.show or (args.out_video_root != '')
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    # cap = cv2.VideoCapture(args.video_path)
    # fps = None

    # assert cap.isOpened(), f'Faild to load video file {args.video_path}'

    if args.out_video_root == '':
        save_out_video = False
    else:
        os.makedirs(args.out_video_root, exist_ok=True)
        save_out_video = True

    # optional
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    next_id = 0
    pose_results = []

    # modifications: declaring paths, fps, size

    # path = r"/content/drive/MyDrive/ML_Oct'21_Task_Videos"
    path = args.video_path
    # path = "D:\\CSE-17\\ML\\Headless Tech\\test_videos"

    vid1 = os.path.join(path, "1.mp4")
    vid2 = os.path.join(path, "2.mp4")
    vid3 = os.path.join(path, "3.mp4")
    vid4 = os.path.join(path, "4.mp4")

    vids = [vid1, vid2, vid3, vid4]

    #get video frame count
    size = (1920,1080)
    fps = 30

    if save_out_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            videoWriter = cv2.VideoWriter(
                os.path.join(args.out_video_root, 'combined_output.mp4'), fourcc, fps, size)

    v1 = vid_fps_duration_frames(vid1)
    v2 = vid_fps_duration_frames(vid1)
    v3 = vid_fps_duration_frames(vid1)
    v4 = vid_fps_duration_frames(vid1)

    # taking frame range from the largest video
    frames_range = max(v1[2], v2[2], v3[2], v4[2])
    frames_range = int(frames_range)

    for frame_no in range(frames_range):
        pose_results_last = pose_results

        # flag, img = cap.read()
        # if not flag:
        #     break

        # modification: getting cimbined frames 
        img = combining_frames(vids, size, frame_no)

        # test a single image, the resulting box is (x1, y1, x2, y2)
        mmdet_results = inference_detector(det_model, img)

        # keep the person class bounding boxes.
        person_results = process_mmdet_results(mmdet_results, args.det_cat_id)

        # test a single image, with a list of bboxes.
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            img,
            person_results,
            bbox_thr=args.bbox_thr,
            format='xyxy',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        # get track id for each person instance
        pose_results, next_id = get_track_id(
            pose_results,
            pose_results_last,
            next_id,
            use_oks=args.use_oks_tracking,
            tracking_thr=args.tracking_thr,
            use_one_euro=args.euro,
            fps=fps)

        # show the results
        vis_img = vis_pose_tracking_result(
            pose_model,
            img,
            pose_results,
            radius=args.radius,
            thickness=args.thickness,
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=args.kpt_thr,
            show=False)

        if args.show:
            cv2.imshow('Image', vis_img)
        
        # modification
        if save_out_video:
            videoWriter.write(vis_img)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # cap.release()
    if save_out_video:
        videoWriter.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
