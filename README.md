# Pose-Detection-From-Vids
To run the code in colab:
- first install the dependencies: 
    - !pip install mmcv-full
    - !rm -rf mmdetection
    - !git clone https://github.com/open-mmlab/mmdetection.git
    - %cd mmdetection
    - !pip install -e .
    - !git clone https://github.com/open-mmlab/mmpose
    - %cd mmpose
    - !pip install -r requirements.txt
    - !pip install -v -e .

- Then upload the python script file in session and run the script as per given in run command.txt. Change the command accordingly for windows/linux/MacOS
    
- Here in the run command, In the first line, main python script location has to be provided. Video-path argument should have the location of the task videos folder and for out-video give the out video location.
- If for some reason input path related error issue is faced , please check 181-182 line of the python script
- To change resolution or fps check 193-194 line of the script
- A working colab code notebook link for testing: https://colab.research.google.com/drive/1XlXisPVbmhtFE8phwECuH4GIgUuvjYiN?usp=sharing
- Final generated video combined_output.mp4 can be found in the colab sessions task_output_video folder. 
