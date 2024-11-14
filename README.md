# Non-invasive eye tracking and rentinal view reconstruction in free swimming schooling fish
This folder contains the source code for performing goldfish body mesh reconstruction and retinal view reconstruction. 
To test the code with our original input data, please download the demo package from the following link:

### Trained model and demo data:
https://drive.google.com/file/d/1mF2jqTQG6ojuwV-hV819c6X8XPzzyIt5/view?usp=sharing

This package contains the source code together with the trained detection model and a pair of input videos.
Follow the instruction here to perform video processing, body mesh reconstruction, and view reconstruction step by step.

### Install dependencies:
The files in the demo package are already placed in the corresponding folders.
Before running the demo script, the python environment must be configured.
We recommend using conda environment to take care of the configuration:
```
conda env create -f fishET.yml
```
Then activate the environment with:
```
conda activate fishET
```
Alternatively, the dependencies can be installed use pip with the requirements.txt file in this repository.

In the end, the PyTorch3D package should be installed:
https://github.com/facebookresearch/pytorch3d

---

In the reconstruction we utilize GPU to accelerate the processing. 
Although, the whole process can be done by CPU at a lower speed as well.
Since the required driver should be installed according to the GPU model, which varies in different PC, we recommend the official guide from PyTorch for the GPU cofiguration:
https://pytorch.org/get-started/locally/

### Execute demo code:
In the fishET environment, run the demo script in the demo folder:
```
python3 fish_4D_demo.py
```
