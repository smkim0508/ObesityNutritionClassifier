**AOS**: Anti-Obesity System to classify unhealthy food
========

TODO LIST
[ ] 설명 실행 예시 그림 추가
- [X] 데이터셋 링크 추가

Dataset for AOS (Anti-Obesity System)

![DETR](./figure/Dataset_snapshot.png)

**What it is**. 
A classification system proposed to combat the never-ending obesity issues around the world. It takes RGB images as input and outputs Nutirional data and healthier alternatives to hopefully deter people from consuming unhealthy foods too often. 

**About the code**. We believe that object detection should not be more difficult than classification,
and should not require complex libraries for training and inference.
DETR is very simple to implement and experiment with, and we provide a
[standalone Colab Notebook](https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_demo.ipynb)
showing how to do inference with DETR in only a few lines of PyTorch code.
Training code follows this idea - it is not a library,
but simply a [main.py](main.py) importing model and criterion
definitions with standard training loops.

Additionnally, we provide a Detectron2 wrapper in the d2/ folder. See the readme there for more information.

For details see [End-to-End Object Detection with Transformers](https://ai.facebook.com/research/publications/end-to-end-object-detection-with-transformers) by Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko.

# Dataset Download
To download the test samples, click
[here](https://drive.google.com/drive/folders/1TIGOFiS9U7x2uX34IuM_OoXFMbmAhIrf?usp=sharing).

To download the train samples, click [here](https://drive.google.com/drive/folders/1KU8HUKFAW_SCy4MNnGBikeBwJLGSLxZK?usp=sharing).

---

# How to Run
*Explain how to train evaluation
```bash
python train.py
```
To test model, run ~~
```
python demo.py --image.path /your/image/path
```
---

# Model Zoo
AOS-RC denotes the AOS model with the new Random Conjunction augmentation method applied. 
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>AP</th>
      <th>url</th>
      <th>size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AOS</td>
      <td>0.59</td>
      <td><a href="https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth">model</a></a></td>
      <td>??Mb</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AOS-RC</td>
      <td>??</td>
      <td><a href="https://dl.fbaipublicfiles.com/detr/detr-r50-dc5-f0fb7ef5.pth">model</a></a></td>
      <td>??Mb</td>
    </tr>
  </tbody>
</table>

--- 


