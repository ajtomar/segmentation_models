## SAM-adapter: Adapting SAM in Underperformed Scenes

Tianrun Chen, Lanyun Zhu, Chaotao Ding, Runlong Cao, Yan Wang, Shangzhan Zhang, Zejian Li, Lingyun Sun, Papa Mao, Ying Zang

<a href='https://www.kokoni3d.com/'> KOKONI, Moxin Technology (Huzhou) Co., LTD </a>, Zhejiang University, Singapore University of Technology and Design, Huzhou University, Beihang University.

In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 3367-3375).

  <a href='https://tianrun-chen.github.io/SAM-Adaptor/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
## 

<a href='https://arxiv.org/abs/2304.09148'><img src='https://img.shields.io/badge/ArXiv-2304.09148-red' /></a> 
Update on 30 August: This paper will be prsented at ICCV 2023. 

Update on 28 April: We tested the performance of polyp segmentation to show our approach can also work on medical datasets.
<img src='https://tianrun-chen.github.io/SAM-Adaptor/static/images/polyp.jpg'>
Update on 22 April: We report our SOTA result based on ViT-H version of SAM (use demo.yaml). We have also uploaded the yaml config for ViT-L and ViT-B version of SAM, suitable  GPU with smaller memory (e.g. NVIDIA Tesla V-100), although they may compromise on accuracy.

## Environment
This code was implemented with Python 3.8 and PyTorch 1.13.0. You can install all the requirements via:
```bash
pip install -r requirements.txt
```


## Quick Start
1. Download the dataset and put it in ./load.
2. Download the pre-trained [SAM(Segment Anything)](https://github.com/facebookresearch/segment-anything) and put it in ./pretrained.
3. Training:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes 1 --nproc_per_node 4 loadddptrain.py --config configs/demo.yaml
```
!Please note that the SAM model consume much memory. We use 4 x A100 graphics card for training. If you encounter the memory issue, please try to use graphics cards with larger memory!


4. Evaluation:
```bash
python test.py --config [CONFIG_PATH] --model [MODEL_PATH]
```

## Train
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch train.py --nnodes 1 --nproc_per_node 4 --config [CONFIG_PATH]
```
Updates on 30 July. As mentioned by @YunyaGaoTree in issue #39
You can also try to use the code below to gain (probably) faster training.
```bash
!torchrun train.py --config configs/demo.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes 1 --nproc_per_node 4 loadddptrain.py --config configs/demo.yaml
```

## Test
```bash
python test.py --config [CONFIG_PATH] --model [MODEL_PATH]
```

## Pre-trained Models
Download the pretrained model for GBC from [here](https://drive.google.com/file/d/1GAjgmvO-EBagvPb8F4W5YCXrMNgyhs-j/view?usp=sharing)

## Dataset

### Camouflaged Object Detection
- **[COD10K](https://github.com/DengPingFan/SINet/)**
- **[CAMO](https://drive.google.com/open?id=1h-OqZdwkuPhBvGcVAwmh0f1NGqlH_4B6)**
- **[CHAMELEON](https://www.polsl.pl/rau6/datasets/)**

### Shadow Detection
- **[ISTD](https://github.com/DeepInsight-PCALab/ST-CGAN)**

### Polyp Segmentation - Medical Applications
- **[Kvasir](https://datasets.simula.no/kvasir-seg/)**

## Citation

If you find our work useful in your research, please consider citing:

```
@misc{chen2023sam,
      title={SAM Fails to Segment Anything? -- SAM-Adapter: Adapting SAM in Underperformed Scenes: Camouflage, Shadow, and More}, 
      author={Tianrun Chen and Lanyun Zhu and Chaotao Ding and Runlong Cao and Shangzhan Zhang and Yan Wang and Zejian Li and Lingyun Sun and Papa Mao and Ying Zang},
      year={2023},
      eprint={2304.09148},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgements
The part of the code is derived from Explicit Visual Prompt   <a href='https://nifangbaage.github.io/Explicit-Visual-Prompt/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> by 
Weihuang Liu, [Xi Shen](https://xishen0220.github.io/), [Chi-Man Pun](https://www.cis.um.edu.mo/~cmpun/), and [Xiaodong Cun](https://vinthony.github.io/) by University of Macau and Tencent AI Lab.

