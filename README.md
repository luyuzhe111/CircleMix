# Fine-grained Glomerular Lesion Classification with Deep Learning

## Relevant Publications:

### [SPIE 2021](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/11596/1159609/Improve-global-glomerulosclerosis-classification-with-imbalanced-data-using-CircleMix-augmentation/10.1117/12.2580482.full?tab=ArticleLink)

> Yuzhe Lu, Haichun Yang, Zheyu Zhu, Ruining Deng, Agnes B. Fogo, and Yuankai Huo "Improve global glomerulosclerosis classification with imbalanced data using CircleMix augmentation", Proc. SPIE 11596, Medical Imaging 2021: Image Processing, 1159609 (15 February 2021).

In this paper, we proposed the CircleMix, a novel data augmentation optimized for for the ball-shaped biomedical objects, such as the glomeruli, to improve the performance of the classifier. The implementation lies in line 207-235 in ```train.py```. A simple illustration is attached below.

<br/>
<p align="center">
  <img src="https://github.com/luyuzhe111/CircleMix/blob/master/renal/demo/demo.png" width="50%">
</p>
<br/>

### [JMI 2022](https://www.spiedigitallibrary.org/journals/journal-of-medical-imaging/volume-9/issue-1/014005/Holistic-fine-grained-global-glomerulosclerosis-characterization--from-detection-to/10.1117/1.JMI.9.1.014005.full)

> Yuzhe Lu, Haichun Yang, Zuhayr Asad, Zheyu Zhu, Tianyuan Yao, Jiachen Xu, Agnes B. Fogo, and Yuankai Huo "Holistic fine-grained global glomerulosclerosis characterization: from detection to unbalanced classification," Journal of Medical Imaging 9(1), 014005 (17 February 2022).

In this paper, we empirically demonstrated the forward transferability of models pretrained on large natural image datasets to medical images, particularly for the task of fine-grained global glomerulosclerosis characterization. We implemented a two-stage pipeline that connects glomeruli detection and classification algorithms to aid renal pathologists in their diagnosis. 

