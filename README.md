# yolov5_ambulance_detection

<div align="center">
<p>
   <a align="left" href="https://ultralytics.com/yolov5" target="_blank">
   <img width="850" src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d2/LJ58OJX_LONDON_AMBULANCE_QUEEN_VICTORIA_STREET_CITY_OF_LONDON_%2826124592264%29.jpg/1920px-LJ58OJX_LONDON_AMBULANCE_QUEEN_VICTORIA_STREET_CITY_OF_LONDON_%2826124592264%29.jpg"></a>
</p>

<br>
<br>
<p>
YOLOv5 🚀 is a family of object detection architectures and models pretrained on the COCO dataset, and represents <a href="https://ultralytics.com">Ultralytics</a>
 open-source research into future vision AI methods, incorporating lessons learned and best practices evolved over thousands of hours of research and development.

This university project uses a adaption of the public project from Ultralytics. Please note that this repo no longer contains the latest version of yolo. For the latest releases go to https://github.com/ultralytics/yolov5.
In this example we are training *british ambulance* vehicles to detect in videomaterial. You can adapt this project for your own needs. 

**Please note:** For better training on large datasets and better inferencing results you use a GPU for this project!
</p>

</div>

<!--
<a align="center" href="https://ultralytics.com/yolov5" target="_blank">
<img width="800" src="https://github.com/ultralytics/yolov5/releases/download/v1.0/banner-api.png"></a>
-->

</div>

## <div align="center">Documentation</div>

See the [YOLOv5 Docs](https://docs.ultralytics.com) for full documentation on training, testing and deployment.

## <div align="center">Prepare your Environment</div>

<details open>
<summary>Install</summary>

Clone repo and install [requirements.txt](https://github.com/Jomeezius/yolov5_ambulance_detection/blob/main/requirements.txt) in a

[**Python>=3.8.0**](https://www.python.org/) environment, including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/).

```bash
git clone https://github.com/Jomeezius/yolov5_ambulance_detection  # clone
cd yolov5
pip install -r requirements.txt  # install
```

</details>

<details open>

<summary>Prepare your Data</summary>

Create a project folder in your directory. There should be all data necessary for you custom detection model.
When you are splitting the projects in different folders, it is much easer to seperate.

In this example we are creating a folder for our british ambulance detection. This example data is a opendata set and is free usable. Consider, that you don´t have copyright problems.
To prepare the yolov5 detection we need prepared datasets for training. For this step it is required to label our british vehicles inside our pictures. For this step there are many tools on the market like..

* roboflow
* labelbox 
* labelimg 
* and so on... 

In this case we prepared the data with roboflow and than we placed the data inside the following directory.

```bash
mkdir british_ambulance  # clone
cd british_ambulance
mkdir test train valid
```

After this step, upload the data to the folders test,train,validate. In each folder you should have a folder with images and labels.

<div align="center">
<p>
   <a align="left" href="https://ultralytics.com/yolov5" target="_blank">
   <img width="850" src="/yolov5_ambulance_detection/labeling.PNG"></a>
</p>



Create a custom yolov5 data file for your training. Here you should define the class you want to detect.
In this case we reuse the truck class. When you want, you can define your own class. Have a look under data/coco.yml as reference. But it is not necessary.

**data.yml**

```yaml

# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
# COCO 2017 dataset http://cocodataset.org by Microsoft
# Example usage: python train.py --data coco.yaml
# parent
# ├── yolov5
# └── datasets
#     └── coco  ← downloads here (20.1 GB)

# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
# path: ../datasets/coco # dataset root dir
# train: train2017.txt # train images (relative to 'path') 118287 images
# val: val2017.txt # val images (relative to 'path') 5000 images
# test: test-dev2017.txt # 20288 of 40670 images, submit to https://competitions.codalab.org/competitions/20794

# Classes
# names:
#   0: person
#   1: bicycle
#   2: car
#   3: motorcycle
#   4: airplane
#   5: bus
#   6: train
#   7: truck
#   8: boat
#   9: traffic light
#   10: fire hydrant
#   11: stop sign
#   12: parking meter
#   13: bench
#   14: bird
#   15: cat
#   16: dog
#   17: horse
#   18: sheep
#   19: cow
#   20: elephant
#   21: bear
#   22: zebra
#   23: giraffe
#   24: backpack
#   25: umbrella
#   26: handbag
#   27: tie
#   28: suitcase
#   29: frisbee
#   30: skis
#   31: snowboard
#   32: sports ball
#   33: kite
#   34: baseball bat
#   35: baseball glove
#   36: skateboard
#   37: surfboard
#   38: tennis racket
#   39: bottle
#   40: wine glass
#   41: cup
#   42: fork
#   43: knife
#   44: spoon
#   45: bowl
#   46: banana
#   47: apple
#   48: sandwich
#   49: orange
#   50: broccoli
#   51: carrot
#   52: hot dog
#   53: pizza
#   54: donut
#   55: cake
#   56: chair
#   57: couch
#   58: potted plant
#   59: bed
#   60: dining table
#   61: toilet
#   62: tv
#   63: laptop
#   64: mouse
#   65: remote
#   66: keyboard
#   67: cell phone
#   68: microwave
#   69: oven
#   70: toaster
#   71: sink
#   72: refrigerator
#   73: book
#   74: clock
#   75: vase
#   76: scissors
#   77: teddy bear
#   78: hair drier
#   79: toothbrush

names:
- truck
nc: 1
train: british_ambulance/train/images
val: british_ambulance/valid/images

# Download script/URL (optional)
# download: |
#   from utils.general import download, Path


#   # Download labels
#   segments = False  # segment or box labels
#   dir = Path(yaml['path'])  # dataset root dir
#   url = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/'
#   urls = [url + ('coco2017labels-segments.zip' if segments else 'coco2017labels.zip')]  # labels
#   download(urls, dir=dir.parent)

#   # Download data
#   urls = ['http://images.cocodataset.org/zips/train2017.zip',  # 19G, 118k images
#           'http://images.cocodataset.org/zips/val2017.zip',  # 1G, 5k images
#           'http://images.cocodataset.org/zips/test2017.zip']  # 7G, 41k images (optional)
#   download(urls, dir=dir / 'images', threads=3)

```


</details>

## <div align="center">Run Training and Inference on custom Dataset</div>

<details>
<summary>Training</summary>

The commands below reproduce YOLOv5 [COCO](https://github.com/ultralytics/yolov5/blob/master/data/scripts/get_coco.sh)
results. [Models](https://github.com/ultralytics/yolov5/tree/master/models)
and [datasets](https://github.com/ultralytics/yolov5/tree/master/data) download automatically from the latest
YOLOv5 [release](https://github.com/ultralytics/yolov5/releases). Training times for YOLOv5n/s/m/l/x are
1/2/4/6/8 days on a V100 GPU ([Multi-GPU](https://github.com/ultralytics/yolov5/issues/475) times faster). Use the
largest `--batch-size` possible, or pass `--batch-size -1` for
YOLOv5 [AutoBatch](https://github.com/ultralytics/yolov5/pull/5092). Batch sizes shown for V100-16GB.

```bash
python train.py --data data.yaml --cfg yolov5n.yaml --weights '' --batch-size 128
                                       yolov5s                                64
                                       yolov5m                                40
                                       yolov5l                                24
                                       yolov5x                                16
```


</details>

<details>
<summary>Inference with detect.py</summary>

`detect.py` runs inference on a variety of sources, downloading [models](https://github.com/ultralytics/yolov5/tree/master/models) automatically from
the latest YOLOv5 [release](https://github.com/ultralytics/yolov5/releases) and saving results to `runs/detect`.

```bash
python detect.py --source 0  # webcam
                          img.jpg  # image
                          vid.mp4  # video
                          path/  # directory
                          path/*.jpg  # glob
                          'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                          'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```

</details>


## <div align="center">Why YOLOv5</div>

- **COCO AP val** denotes mAP@0.5:0.95 metric measured on the 5000-image [COCO val2017](http://cocodataset.org) dataset over various inference sizes from 256 to 1536.
- **GPU Speed** measures average inference time per image on [COCO val2017](http://cocodataset.org) dataset using a [AWS p3.2xlarge](https://aws.amazon.com/ec2/instance-types/p3/) V100 instance at batch-size 32.
- **EfficientDet** data from [google/automl](https://github.com/google/automl) at batch size 8.
- **Reproduce** by `python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n6.pt yolov5s6.pt yolov5m6.pt yolov5l6.pt yolov5x6.pt`

</details>


<br>

## <div align="center">Contribute</div>

I love your input! Please send me your feedback on your experiences. Thank you!.

Mathias Frink
Data & Analytics DevOp Consultant