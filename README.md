**AOS**: Anti-Obesity System to classify unhealthy food
========

Dataset for AOS (Anti-Obesity System)

![AOS](./figure/Dataset_snapshot.png)

**What it is**. 
A classification system proposed to combat the never-ending obesity issues around the world. It takes RGB images as input and outputs Nutirional data and healthier alternatives to hopefully deter people from consuming unhealthy foods too often. 

# Dataset Download
To download the test samples, click
[here](https://drive.google.com/drive/folders/1TIGOFiS9U7x2uX34IuM_OoXFMbmAhIrf?usp=sharing).

To download the train samples, click [here](https://drive.google.com/drive/folders/1KU8HUKFAW_SCy4MNnGBikeBwJLGSLxZK?usp=sharing).

---

# How to Run
To train AOS model, run as follows:
```bash
python3 train.py
```
To test AOS model, run as follows:
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
      <td>59.07</td>
      <td><a href="https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth">model</a></a></td>
      <td>??Mb</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AOS-RC</td>
      <td>59.30</td>
      <td><a href="https://dl.fbaipublicfiles.com/detr/detr-r50-dc5-f0fb7ef5.pth">model</a></a></td>
      <td>??Mb</td>
    </tr>
  </tbody>
</table>

--- 


