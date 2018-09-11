## Reid-mask

---

This repo investigates how to utilize `mask` data in reid task. Several strategies are taken in our experiments, and evaluated on [Market-1501 dataset](http://www.liangzheng.org/Project/project_reid.html). Our baseline is based on a strong baseline [here](https://github.com/L1aoXingyu/reid_baseline), and use Resnet50 as base model.

Strategies of applying `mask` data:

* Concatenate mask and RGB image to form a new image with 4 channels.
* Soft and hard attention proposed in [MGCAM](http://openaccess.thecvf.com/content_cvpr_2018/papers/Song_Mask-Guided_Contrastive_Attention_CVPR_2018_paper.pdf).
* Spatial and channel attention proposed in [CBAM](https://arxiv.org/pdf/1807.06521.pdf).


### Preparations

---

Before starting running this code, you should make the following preparations:

* Download the [Market-1501 dataset](http://www.liangzheng.org/Project/project_reid.html).
* The mask data are avaliable [here](https://github.com/developfeng/mgcam).
* Install MXNet. This repository is tested on official [MXNet v1.3.0](https://github.com/apache/incubator-mxnet).


### experiments

---

* Modify related settings in `.yml` files first, and train the model:
```shell
python train.py/train-mask.py/train-cbam-att.py
```

These three files corresponding to experiments: reid baseline, 4-channel soft/hard attention and spatial/channel attention. More details can be found in the files and their `.yml` files.

* Then use `eval.py` to extract features for specific testing set and evaluate the models.

* R1 performance with RGB/RGBM input and soft/hard attention

| |baseline|soft mask|hard mask|
|---|---|---|---|
|RGB|91.1|90.9|90.9|
|RGBM|92.4|92.6|92.6|


* R1 performance with spatail and channel attention

| | baseline | channel | spatial | spatial+channel |
|---|---|---|---|---|
|RGB| 91.1 | 91.5 | 90.3 | 91.8 |
|RGBM| 92.4 | 93.6 | 91.2 | 92.4 |

#### note

* This repo also include the codes for evaluating the occlusion in reid task, i.e. `eval_verify.py`, and related list processing files in `utils` dir.
* The attention map of RGB-soft mask model are displayed below. Four image are taken as a group, in which they are arranged as RGB original image, GCAM visual map, attention map and mask ground truth.

![attention map](vis/soft-mask.png = 250x250)
