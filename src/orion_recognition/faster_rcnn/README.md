## Object detector

We use Faster R-CNN implemented in PyTorch for object detection and bounding box prediction. 
We finetune Faster R-CNN on the RoboCup@Home objects dataset to come up with a better object prediction.

### Requirements
```bash
pip3 install torch torchvision torchaudio einops matplotlib pycocotools
```


### Helpful resources

- [Official PyTorch tutorial on finetuning Faster R-CNN](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
- https://www.kaggle.com/code/yerramvarun/fine-tuning-faster-rcnn-using-pytorch/notebook
- https://haochen23.github.io/2020/06/fine-tune-faster-rcnn-pytorch.html#.YpDWRznMJH6