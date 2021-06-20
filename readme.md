# Sketch Generation

## Description

The goal of this project is to provide enough data sets for high-resolution sketch completion, the work is to convert natural world pictures into a variety of high-resolution sketches (vector format). Finally, multiple sets of "*world pictures* vs *sketches*" image pairs are obtained. In the mean time, a GUI interface is also provided to filter pictures of different quality.

<center class='half'>
    <img src="docs/motorbike1.png" width="700">
    <img src="docs/motorbike.gif" width="250">
</center>

<center class='half'>
    <img src="docs/baby1.png" width="700">
    <img src="docs/baby.gif" width="250">
</center>

**Description (from left to right):** (1) World image (2) Pencil style image (3) Clean line sketch, pixel format (4) Stroke sketch, vector format



## Dependencies

Only some important dependencies are list here

* Tensorflow (1.12.0 <= version <= 1.15.0)

* PyTorch 

* OpenCV

* gizeh

  

## Pipeline

### 1) Image 2 Pencil

[python](image2pencil/pencilize.py)

