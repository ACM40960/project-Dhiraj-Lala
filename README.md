# Moving Object detection and tracking with Bird's Eye View Simulation

## Table of Contents 📑
- [Overview](##Overview)
- [YOLOv5 Overview](##YOLOv5-Overview)
- [Requirements](##Requirements)
- [Installation](##Installation)
- [Usage/Examples](##Usage/Examples)
- [Customization](##Customization)
- [Initialization](##Initialization)
- [Video Processing](##Video-Processing)
- [Post Processing and Visualization](##Post-Processing-and-Visualization)
- [Cleanup](##Cleanup)
- [Result](##Result)
- [Acknowledgements](##Acknowledgements)
- [Authors](##Authors)


## Overview 🎯

This Python application simulates the tracked items from a bird's eye view perspective while demonstrating object identification and tracking using YOLOv5. The script analyzes video input and superimposes virtual objects onto the aerial view to better understand object movement and interaction.

## YOLOv5 Overview 🛣️

YOLOv5 is a single-shot detector. It means it requires only one forward pass through the network to detect objects.The model uses predefined bounding box shapes called anchors, which are optimized for the dataset at hand. These anchors help in predicting the sizes of the detected objects.

## Requirements 💽

- Python 3.x
- OpenCV (OpenCV Version: 4.8.0)
- NumPy (NumPy Version: 1.21.5)
- PyTorch (PyTorch Version: 2.0.1+cpu)
- torchvision (Torchvision Version: 0.15.2+cpu)
- Matplotlib (Version: 3.5.1)
- [YOLOv5](https://github.com/ultralytics/yolov5)


## Installation 📥

1. **Clone the repository**:

```bash
git clone [https://github.com/your-username/your-repo](https://github.com/ACM40960/project-dwaghaye).git
cd your-repo

```

2. **Install the required packages**:

```bash
pip install opencv-python==4.8.0
pip install numpy==1.21.5
pip install torch==2.0.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install torchvision==0.15.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install matplotlib==3.5.1

```

3. **Jupyter Notebook**: To run the provided Jupyter Notebook, install and start Jupyter:

```bash
pip install jupyter
jupyter notebook

```

## Usage/Examples 🚀

1. Place the video file you want to process in the repository directory.

2. Update the path_input_video (with the extension .mov) and path_output_video (with the extension .mp4) variables in the script with the path to your input video.
3. Download the picture MyCar.png' from the repository and put the path to the image in your local system in the overlay_img variable.

4. Run the script.

## Customization 🛠️

The script can be modified by changing the following variables:

- path_input_video: You can select a different path to play a different video.
- path_output_video: Path to save the output video file.
- yolo_classes: List of YOLO class names. These can be added or removed based on requirement.
- enable_track: If set to True, object tracking is enabled; if set to False, it is not.
- size_of_batch: Batch size for YOLO inference.

## Initialization 👾

- YOLOv5 model is loaded.
- Video paths, settings, and class labels are initialized.
- Helper functions are defined:
  
        - transparent_overlay_check: Overlays an image with transparency onto another image.
        - simulation_of_object: Simulates an object on a given background using the class name.
        - overlay_car: Overlays a car image on a background.
        - object_bev_plot: Projects object centroids to a bird's eye view.
        - tracking_and_distance_details: Tracks objects between frames based on centroid proximity.

## Video Processing 📽️

- Each frame of the video is read.
- YOLOv5 is used for object detection.
- Objects are filtered based on class and confidence score.
- The detected objects are tracked between frames using a simple distance-based approach.
- A bird's eye view transformation is applied to object centroids.
- Detected objects are simulated and plotted from a bird's eye view perspective.
- The original frame, the simulated objects, and the transformed bird's eye view are displayed.
- The processing time for each frame is recorded.

## Post Processing and Visualization 🎞️

- Processing times per frame are plotted.
- Detected and tracked object counts are plotted.

## Cleanup 🌪️

- Video input and output streams are released. Any open windows are destroyed.

## Result 📋

A real-time presentation of the original video, complete with object detection bounding boxes, class, and probability, as well as a simulated bird's-eye view with superimposed objects and a converted bird's-eye view with annotated centroids.


<img width="300" height="200" alt="Screenshot 2023-08-17 123651" src="https://github.com/ACM40960/project-dwaghaye/assets/118671691/f9b9eee9-9e53-466e-b440-0db75132fd5c">


<img src="https://github.com/ACM40960/project-dwaghaye/assets/118671691/f7a1a249-0260-4bd1-9f96-2d510d2e7f0d" width="300" height="200">

Bird's Eye View:

<img src="https://github.com/ACM40960/project-dwaghaye/assets/118671691/1887b1d1-51e1-4b41-afe4-1a4d89e1a268" width="300" height="200">


## Acknowledgements 📝

- The YOLOv5 model from the https://github.com/ultralytics/yolov5 repository is used in this script. 
- Based on computer vision concepts, approaches for object simulation and bird's-eye view modification are built.

## Authors 👩‍💻👨‍💻

Dhaarna Waghaye (22201256)

Dhiraj Lala (22200259)
