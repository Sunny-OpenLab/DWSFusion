# DWSFusion
This is official Pytorch implementation of "DWSFusion: A Lightweight Infrared and Visible Image Fusion Method Based on Dual Weight Supervision and Frequency Adaptive Attention"
## Abstract
Infrared and visible image fusion (IVIF) aims to integrate complementary information from different modalities into a comprehensive image. Existing approaches typically rely on large-scale networks to preserve as much information from the source images as possible, overlooking the adverse impact of redundant information on the fusion outcome. Moreover, balancing the receptive field with the sensitivity to high-frequency details during feature extraction remains a critical challenge. To address these issues, this paper proposes DWSFusion, a lightweight infrared and visible image fusion method based on dual-weight supervision and frequency-adaptive attention. Firstly, a weight estimation module, grounded in a frequency-adaptive attention mechanism, is designed to extract weight maps from the source images. Next, a cross-weight strategy is introduced to retain essential information from the source images while suppressing redundancy. Subsequently, the feature maps are input into a frequency-adaptive feature fusion module to generate the fused image. An adaptive weight estimation module is employed to extract multimodal weight maps of the fused image, and a dual-weight supervision strategy at the feature level is proposed to enhance the feature extraction capacity of the adaptive feature extraction module and the feature fusion capacity of the adaptive fusion module. Finally, a dual discriminator based on the WGAN-GP structure is utilized to further encourage the generative network to preserve critical information from the source images. Extensive experimental results demonstrate that the proposed method outperforms existing fusion models in terms of subjective visual quality and quantitative evaluation metrics, while significantly reducing model parameter size.
## Highlight
- We propose a novel method based on dual-weight supervision and frequency-adaptive attention network for IR/VIS fusion.
- It can keep both the complementary information highlighted and the redundant information suppressed.
- It can balance the receptive field range and the attention to high-frequency features of the image.
- A feature-level dual-weight supervision strategy and a MS-SSIM loss function are designed.
- It is an end-to-end model that requires an exceptionally small number of parameters.
  
![Comprehensive.png](Figure%2FComprehensive.png)  
## Framework
![DWS.png](Figure%2FDWS.png)
![CW.png](Figure%2FCW.png)  
The framework of the proposed dual-weight supervision strategy at the feature level and cross-weight strategy.
## Network Architecture
![Network.png](Figure%2FNetwork.png)  
The architecture of the DWSFusion.
## Code Usage
### Environment
```pip install -r requirements```
### To Train
Run ```python main.py``` to train your model. The training data is obtained by extracting patches from the images in the MSRS dataset.
For convenient training, users can download the training dataset from [here](https://pan.baidu.com/s/16qbgI3HK7Y45H0GwZkc8vQ?pwd=Qi42), in which the extraction code is: Qi42.
Put this tar file into folder data.
### To Test
Run ```python test.py``` to test the model.  
M3FD dataset can be downloaded from [M3FD](https://pan.baidu.com/s/1SbqLk2YSAYr_1NKVCIeNsQ?pwd=Qi42), in which the extraction code is: Qi42.  
Put this tar file into folder data/test_data.
### Recommended Environment
- torch==1.11.0+cu113
- torchvision==0.12.0+cu113
- numpy==1.26.4
- opencv-python==4.10.0.84
- mmcv-full==1.5.3

## Fusion Example
Qualitative comparison of DWSFusion with 18 state-of-the-art methods from the TNO, RoadScene, MSRS and M3FD datasets.
### TNO  
![TNO.png](Figure%2FTNO.png)
### RoadScene
![RoadScene.png](Figure%2FRoadScene.png)
### MSRS  
![MSRS.png](Figure%2FMSRS.png)
### M3FD  
![M3FD.png](Figure%2FM3FD.png)

## Detection Results
Detection results for infrared, visible and fused images from the MSRS dataset. The segmentation model is YOLOv5s.  
![Detection.png](Figure%2FDetection.png)

## Segmentation Results
Segmentation results for infrared, visible and fused images from the MSRS dataset. The segmentation model is Deeplabv3+, pre-trained on the Cityscapes dataset.
![Segmentation.png](Figure%2FSegmentation.png)

## If this work is helpful to you, please cite it as：
```

```