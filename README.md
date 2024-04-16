# Image Super-Resolution via Iterative Refinement

[Paper](https://arxiv.org/pdf/2104.07636.pdf ) |  [Project](https://iterative-refinement.github.io/ )

## Brief

This is an unofficial implementation of **Image Super-Resolution via Iterative Refinement(SR3)** by **PyTorch**.

There are some implementation details that may vary from the paper's description, which may be different from the actual `SR3` structure due to details missing. Specifically, we:

- Used the ResNet block and channel concatenation style like vanilla `DDPM`.
- Used the attention mechanism in low-resolution features like vanilla `DDPM`.
- Encode the $\gamma$ as `FilM` structure did in `WaveGrad`, and embed it without affine transformation.
- Define the posterior variance as $\dfrac{1-\gamma_{t-1}}{1-\gamma_{t}} \beta_t$  rather than $\beta_t$,  which gives similar results to the vanilla paper.


### Conditional Generation (with Super Resolution)

- [x] 114×114 -> 480x480 on FFHQ-CelebaHQ


### Training Step

- [x] log / logger
- [x] metrics evaluation
- [x] multi-gpu support
- [x] resume training / pretrained model
- [x] validate alone script



### Data Prepare

#### New Start

Download the dataset and prepare it in **LMDB** or **PNG** format using script.

```python
# Resize to get 114x114 LR_IMGS and 480x480 HR_IMGS, then prepare 128×128 Fake SR_IMGS by bicubic interpolation
python data/prepare_data.py  --path [dataset root]  --out [output root] --size 114x480 -l
```

then you need to change the datasets config to your data path and image resolution: 

```json
"datasets": {
    "train": {
        "dataroot": "dataset/ffhq_16_128", // [output root] in prepare.py script
        "l_resolution": 114, // low resolution need to super_resolution
        "r_resolution": 480, // high resolution
        "datatype": "lmdb", //lmdb or img, path of img files
    },
    "val": {
        "dataroot": "dataset/celebahq_16_128", // [output root] in prepare.py script
    }
},
```

#### Own Data

You also can use your image data by following steps, and we have some examples in dataset folder.

At first, you should organize the images layout like this, this step can be finished by `data/prepare_data.py` automatically:

```shell
# set the high/low resolution images, bicubic interpolation images path 
dataset/celebahq_16_128/
├── hr_480 # it's same with sr_16_128 directory if you don't have ground-truth images.
├── lr_114 # vinilla low resolution images
└── sr_114_480 # images ready to super resolution
```

```python
# super resolution from 114 to 480
python data/prepare_data.py  --path [dataset root]  --out celebahq --size 114,480 -l
```

*Note: Above script can be used whether you have the vanilla high-resolution images or not.*

then you need to change the dataset config to your data path and image resolution: 

```json
"datasets": {
    "train|val": { // train and validation part
        "dataroot": "dataset/celebahq_16_128",
        "l_resolution": 114, // low resolution need to super_resolution
        "r_resolution": 480, // high resolution
        "datatype": "img", //lmdb or img, path of img files
    }
},
```

### Training/Resume Training

```python
# Use sr.py and sample.py to train the super resolution task and unconditional generation task, respectively.
# Edit json files to adjust network structure and hyperparameters
python sr.py -p train -c config/sr_sr3.json
```

### Test/Evaluation

```python
# Edit json to add pretrain model path and run the evaluation 
python sr.py -p val -c config/sr_sr3.json

# Quantitative evaluation alone using SSIM/PSNR metrics on given result root
python eval.py -p [result root]
```

### Inference Alone

Set the  image path like steps in `Own Data`, then run the script:

```python
# run the script
python infer.py -c [config file]

