# tree_location
This repository is used in paper: Novel Method of Fruit Tree Trunk Location Using Stereo Camera and Semantic Segmentation by HanSun.
=======
## Novel Method of Fruit Tree Trunk Detection and Location Using Stereo Camera and Semantic Segmentation
---
This is the code repository used in the paper. It includes using the ZED2i camera to obtain relevant data, using the improved model to detect the tree trunk, and measuring the distance of the tree trunk.

## Instructions

### Requirements
	pyzed==3.8
	scipy==1.2.1
	numpy==1.17.0
	matplotlib==3.1.2
	opencv_python==4.1.2.30
	torch==1.2.0
	torchvision==0.4.0
	tqdm==4.60.0
	Pillow==8.2.0
	h5py==2.10.0

Run script
```Python
python /main_script.py
```

Follow the prompts to turn on the camera and perform detection and ranging. The automated program automatically counts each ranging, outputs and saves the results.

### Save route
	The ranging data is saved in "/npy" in the form of .npy
	Visualized data is saved in "/chart"
	The depth image output by the camera is saved in "/deepmap"
	The recognition image output by the algorithm is saved in "/img_out"
	Logs are saved in "/logs"

### Thank you for reviewing this manuscript.

### Reference
https://github.com/ggyyzm/pytorch_segmentation
https://github.com/bubbliiiing/pspnet-pytorch
https://github.com/wkentaro/labelme
