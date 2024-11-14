# Non-invasive eye tracking and rentinal view reconstruction in free swimming schooling fish
This folder contains the source code for goldfish body mesh reconstruction and retinal view reconstruction.
To test the code using our original input data, please download the demo package from the following link:

### Trained model and demo data:
https://drive.google.com/file/d/1mF2jqTQG6ojuwV-hV819c6X8XPzzyIt5/view?usp=sharing
\
This package includes the source code, the trained detection model, and a pair of input videos.
Follow the instructions here to perform video processing, body mesh reconstruction, and view reconstruction step by step.

### Install dependencies:
The files in the demo package are already organized in their corresponding folders.
Before running the demo script, the Python environment must be configured.
We recommend using a Conda environment for managing the configuration:
```
conda env create -f fishET.yml
```
Then activate the environment with:
```
conda activate fishET
```
Alternatively, the dependencies can be installed using `pip` with the `requirements.txt` file in this repository.

Finally, the PyTorch3D package should also be installed:
\
https://github.com/facebookresearch/pytorch3d

---

In the reconstruction process, we utilize the GPU to accelerate processing.  
However, the entire process can also be performed using the CPU, albeit at a slower speed.  
Since the required driver must be installed according to the GPU model, which varies between different PCs, we recommend referring to the official PyTorch guide for GPU configuration:
\
https://pytorch.org/get-started/locally/

### Execute demo code:
In the fishET environment, run the demo script in the demo folder:
```
python3 fish_4D_demo.py
```
