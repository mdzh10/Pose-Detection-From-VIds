# Pose-Detection-From-Vids
To run the code in colab:
- first install the dependencies: 
!pip install mmcv-full

# Install mmdetection
!rm -rf mmdetection
!git clone https://github.com/open-mmlab/mmdetection.git
%cd mmdetection

!pip install -e .

!git clone https://github.com/open-mmlab/mmpose
%cd mmpose

!pip install -r requirements.txt
!pip install -v -e .

- The upload the script file in session and run the following script:
!python /content/top_down_pose_tracking_demo_with_mmdet.py \
    /content/mmdetection/mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
    https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res50_coco_256x192.py \
    https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_256x192-ec54d7f3_20200709.pth \
    --video-path /content/drive/MyDrive/ML_Oct"'"21_Task_Videos \
    --out-video-root /content/task_output_video
    
Here in the run command, In the first line, main python script location has to be provided. Video-path argument should have the location of the task videos folder and for out-video give the out video location.
