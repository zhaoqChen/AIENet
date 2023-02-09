# All-in-One Aerial Image Enhancement Network for  Forest Scenes

Zhaoqi Chen, Chuansheng Wang, Fuquan Zhang, Antoni Grau, and Edmundo Guerra

<hr />

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
Datasets used by AIENet can be downloaded from link: [https://pan.baidu.com/s/1YYXEikK2UVh-kMur0khacA](https://pan.baidu.com/s/1YYXEikK2UVh-kMur0khacA) with code 6mg9.

To train the models of image dehazy, image motion deblur and image compression deblur on your own images, run 
```
python train.py --task task_name --gpu_id gpu_id --indir input_directory --outdir output_directory
```

For testing, please run:
```
python test.py --task task_name --gpu_id gpu_id --indir input_directory --outdir output_directory
```

## Results
Experiments are performed for different image enhancement tasks including, image dehazing, image motion deblurring and image compression deblurring.

<p align="center"><img src = "https://github.com/zhaoqChen/AIENet/blob/main/imgs/quantitative comparisons.png" width="700"></p>

## Contact
Should you have any question, please contact zhaoq_chen@163.com
