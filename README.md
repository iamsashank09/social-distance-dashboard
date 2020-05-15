
# Social Distance Dashboard - Recognize and Visualize risk

ADDITONAL PROJECT: [Mask Detection and Classification \[Code and Demo\]](https://github.com/iamsashank09/mask-detection-and-classification)

![enter image description here](https://github.com/iamsashank09/mask-detection-and-classification/blob/master/outputs/output_P1.gif)

# 
In a world recovering from the shock of COVID-19, it has become increasingly important to practice social distancing. A deep understanding of the pandemic shows how one person's negligence can cause widespread harm which will be hard to negate. So, using Computer Vision and Deep Learning might help monitor the practice of social distancing. 

The application detects people who are close by and not adhering to the distancing norms and marks them in RED bounding boxes, signifying risk. Others, are in GREEN.

The Dashboard at the right, gives a visual representation of the data.
The number next to the GREEN and RED icons are the number of SAFE and RISK people. Whereas, the BLACK is the total number of people in the frame. 

The Pie Chart at the top just plots the SAFE vs AT RISK persons in the frame. 

### Here is a demo containing the application output: 

![Social Distance Dashboard output in Outdoor conditions](https://github.com/iamsashank09/social-distance-dashboard/blob/master/OutputVideos/output-outdoorlong.gif)

![Social Distance Dashboard output in Indoor conditions](https://github.com/iamsashank09/social-distance-dashboard/blob/master/OutputVideos/output-indoor.gif)

#### Full output on Youtube:
Click to go to the Youtube Video
[![Social Distance Dashboard](https://github.com/iamsashank09/social-distance-dashboard/blob/master/OutputVideos/output-outdoor.gif)](https://youtu.be/PUTKKx8GBYg)

### How to use the application:
The entire application is encapsulated in the SocialDistanceDashboard.py file, which can be run using the command:

    python3 socialDistanceDashboard.py

#### Input file:
The input can be a video file and needs to be updated on line number 198:

    filename = "videos/video_1.mp4" # Your file path here.

#### YOLO v3 Dependency:
YOLOv3 trained on COCO is used for person recognition and detection, I have not included the weights as a part of the repository as it is quite big (236MB). 
You can download it at https://pjreddie.com/media/files/yolov3.weights
and add it to the folder - "yolo-coco" as a part of the repository. 
I have mentioned the folder path on line number 203, and can be changed.

    yolopath = "yolo-coco/" # Path points to folder containing weights, cfg and names.

### Concerns over such monitoring applications:
While working on this project, I started having concerns over how this kind of monitoring can be used to create spaces with no freedom, which is scary. I understand this kind of work can be used to boost authoritarianism and force suppression. I can only hope we always keep in mind the freedoms of the individual while we build a safe society.  


### Resources:
Object Detection using YOLO from PyImageSearch - [Link to article](https://www.pyimagesearch.com/2018/05/14/a-gentle-guide-to-deep-learning-object-detection/)

Input videos used for testing and development:

Outdoor videos: [VIRAT Dataset](https://viratdata.org/)

Indoor videos: [Learning Recognition Surveillance Group](https://www.tugraz.at/institutes/icg/research/team-bischof/learning-recognition-surveillance/)

![MIT License Social Distance Dashboard](https://camo.githubusercontent.com/2e380136948ff516fd2a276030a07dbc207c9df8/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f6c6963656e73652f444156466f756e646174696f6e2f6361707461696e2d6e336d302e7376673f7374796c653d666c61742d737175617265)







# Social Distance Dashboard - Recognize and Visualize risk

In a world recovering from the shock of COVID-19, it has become increasingly important to practice social distancing. A deep understanding of the pandemic shows how one person's negligence can cause widespread harm which will be hard to negate. So, using Computer Vision and Deep Learning might help monitor the practice of social distancing. 

The application detects people who are close by and not adhering to the distancing norms and marks them in RED bounding boxes, signifying risk. Others, are in GREEN.

The Dashboard at the right, gives a visual representation of the data.
The number next to the GREEN and RED icons are the number of SAFE and RISK people. Whereas, the BLACK is the total number of people in the frame. 

The Pie Chart at the top just plots the SAFE vs AT RISK persons in the frame. 

### Here is a demo containing the application output: 

![Social Distance Dashboard output in Outdoor conditions](https://github.com/iamsashank09/social-distance-dashboard/blob/master/OutputVideos/output-outdoorlong.gif)

![Social Distance Dashboard output in Indoor conditions](https://github.com/iamsashank09/social-distance-dashboard/blob/master/OutputVideos/output-indoor.gif)

#### Full output on Youtube:
Click to go to the Youtube Video
[![Social Distance Dashboard](https://github.com/iamsashank09/social-distance-dashboard/blob/master/OutputVideos/output-outdoor.gif)](https://youtu.be/PUTKKx8GBYg)

### How to use the application:
The entire application is encapsulated in the SocialDistanceDashboard.py file, which can be run using the command:

    python3 socialDistanceDashboard.py

#### Input file:
The input can be a video file and needs to be updated on line number 198:

    filename = "videos/video_1.mp4" # Your file path here.

#### YOLO v3 Dependency:
YOLOv3 trained on COCO is used for person recognition and detection, I have not included the weights as a part of the repository as it is quite big (236MB). 
You can download it at https://pjreddie.com/media/files/yolov3.weights
and add it to the folder - "yolo-coco" as a part of the repository. 
I have mentioned the folder path on line number 203, and can be changed.

    yolopath = "yolo-coco/" # Path points to folder containing weights, cfg and names.

### Concerns over such monitoring applications:
While working on this project, I started having concerns over how this kind of monitoring can be used to create spaces with no freedom, which is scary. I understand this kind of work can be used to boost authoritarianism and force suppression. I can only hope we always keep in mind the freedoms of the individual while we build a safe society.  


### Resources:
Object Detection using YOLO from PyImageSearch - [Link to article](https://www.pyimagesearch.com/2018/05/14/a-gentle-guide-to-deep-learning-object-detection/)

Input videos used for testing and development:

Outdoor videos: [VIRAT Dataset](https://viratdata.org/)

Indoor videos: [Learning Recognition Surveillance Group](https://www.tugraz.at/institutes/icg/research/team-bischof/learning-recognition-surveillance/)

![MIT License Social Distance Dashboard](https://camo.githubusercontent.com/2e380136948ff516fd2a276030a07dbc207c9df8/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f6c6963656e73652f444156466f756e646174696f6e2f6361707461696e2d6e336d302e7376673f7374796c653d666c61742d737175617265)
