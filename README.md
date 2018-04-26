# Dimensional-Emotion-Analysis-of-Facial-Expression

[project page](http://www.ics.uci.edu/~skong2/DimensionalEmotionModel.html "dimensional-emotion-analysis")
[video demo](https://www.youtube.com/watch?v=tVYW9hxgCho&feature=youtu.be "video demo")


![alt text](http://www.ics.uci.edu/~skong2/image2/splashFigure2_icon.png "visualization")


Automated facial expression analysis has a variety of applications in human-computer interaction. Traditional methods mainly analyze prototypical facial expressions of no more than eight discrete emotions as a classification task. However, in practice, spontaneous facial expressions in naturalistic environment can represent not only a wide range of emotions, but also different intensities within an emotion family. In such situation, these methods are not reliable or adequate. In this paper, we propose to train deep convolutional neural networks (CNNs) to analyze facial expressions explainable in a dimensional emotion model. The proposed method accommodates not only a set of basic emotion expressions, but also a full range of other emotions and subtle emotion intensities that we both feel in ourselves and perceive in others in our daily life. Specifically, we first mapped facial expressions into dimensional measures so that we transformed facial expression analysis from a classification problem to a regression one. We then tested our CNN-based methods for facial expression regression and these methods demonstrated promising performance. Moreover, we improved our method by a bilinear pooling which encodes second-order statistics of features. We showed such bilinear-CNN models significantly outperformed their respective baselines. 


**Keywords**  Dimensional Emotion Model, Fine-Grained Analysis, Facial Expression, High-Order Correlation, Psychology, Affective-Cognitive Computing, Physiological Computing.

Please download those models from the [google drive](https://drive.google.com/drive/folders/1i9Qu2FaJ8HtezS1BzwmOaLa5dTxeWw63?usp=sharing).





MatConvNet is used in our project, and some functions are changed/added. Please compile accordingly by adjusting the path --

```python
LD_LIBRARY_PATH=/usr/local/cuda/lib64:local matlab 

path_to_matconvnet = './libs/matconvnet-1.0-beta23_modifiedDagnn/';
run(fullfile(path_to_matconvnet, 'matlab', 'vl_setupnn'));
addpath(fullfile(path_to_matconvnet, 'matlab'));
vl_compilenn('enableGpu', true, ...
               'cudaRoot', '/usr/local/cuda', ...
               'cudaMethod', 'nvcc', ...
               'enableCudnn', true, ...
               'cudnnRoot', '/usr/local/cuda/cudnn/lib64') ;

```



last update: 04/26/2018

Shu Kong

aimerykong At g-m-a-i-l dot com
