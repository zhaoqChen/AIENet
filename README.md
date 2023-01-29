# All-in-One Aerial Image Enhancement Network for  Forest Scenes

Zhaoqi Chen, Chuansheng Wang, Fuquan Zhang, and Antoni Grau

<hr />

> **Abstract:** *Drone monitoring plays an irreplaceable and vital role in forest firefighting because of its wide-range observation angle and real-time transmission of fire characteristics. However, aerial images are often prone to different degradation problems before visual detection. Most current image enhancement methods aim to restore images with specific degradation types. But these methods lack certain generalizability to different degradations, and they are challenging to meet the actual application requirements. Thus, a single model is urgently needed in forest fire monitoring to address the aerial image degradation caused by various common factors such as adverse weather or drone vibration. This paper is dedicated to building an all-in-one framework for various aerial image degradation problems. To this end, we propose an All-in-one Image Enhancement Network(AIENet), which could recover various degraded images in one network. Specifically, we design a new multi-scale receptive field image enhancement block, which can better reconstruct high-resolution details of target regions of different sizes. In particular, this plug-and-play network design enables it to be embedded in any deep learning model. And it has better flexibility and generalization in practical applications. Taking three challenging image enhancement tasks encountered in drone monitoring as examples, we conduct task-specific and all-in-one image enhancement experiments on a synthetic forest wildfire smoke dataset. The results show that the proposed AIENet outperforms or approaches SOTAs quantitatively and qualitatively. Furthermore, in order to prove the effectiveness of the model in advanced vision tasks, we further apply its results to the forest fire detection task, and the detection accuracy has also been significantly improved.* 

## Network Architecture
<table>
  <tr>
    <td> <img src = "https://github.com/zhaoqChen/AIENet/blob/main/imgs/model.png" width="500"> </td>
    <td> <img src = "https://github.com/zhaoqChen/AIENet/blob/main/imgs/block.png" width="400"> </td>
  </tr>
  <tr>
    <td><p align="center"><b>Overall Framework of AIENet</b></p></td>
    <td><p align="center"><b>MRF Enhancement Block</b></p></td>
  </tr>
</table>

## Installation
The model is built in PyTorch 1.8.1 and tested on Ubuntu 20.04 environment (Python3.8, CUDA11.7).

For installing, follow these intructions
```
conda create -n torch1.8 python=3.8
conda activate torch1.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install matplotlib scikit-image opencv-python tqdm
```

## Training and Evaluation

To train the pre-trained models of image dehazy, image motion deblur and image compression deblur on your own images, run 
```
python train.py --task Task_Name --gpu_id gpu_id --indir input_directory --outdir output_directory
```

For testing, please run:
```
python test.py --task [dehaze | derain] --gpu_id [gpu_id] --indir [input directory] --outdir [output directory]
```

## Results
Experiments are performed for different image processing tasks including, image deblurring, image deraining and image denoising.

<p align="center"><img src = "https://github.com/zhaoqChen/AIENet/blob/main/imgs/quantitative comparisons.png" width="700"></p>

## Contact
Should you have any question, please contact zhaoq_chen@163.com
