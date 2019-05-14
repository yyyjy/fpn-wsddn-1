fpn-wsddn Pytorch implementation of Feature Pyramid Network (FPN) plus wsddn, which trained with selectivesearch proposals instead of RPN for weakly supervised Object Detection

fpn is made up of fpn and fasterrcnn whose proposals are generated from rpn, i change it to fpn+selectivesearch+wsddn to complete wsod task.

## Introduction

This project inherits the property of our [pytorch implementation of faster r-cnn](https://github.com/jwyang/faster-rcnn.pytorch). Hence, it also has the following unique features:

* **It is pure Pytorch code**. We convert all the numpy implementations to pytorch.

* **It supports trainig batchsize > 1**. We revise all the layers, including dataloader, roi-pooling, etc., to train with multiple images at each iteration.

* **It supports multiple GPUs**. We use a multiple GPU wrapper (nn.DataParallel here) to make it flexible to use one or more GPUs, as a merit of the above two features.

* **It supports three pooling methods**. We integrate three pooling methods: roi pooing, roi align and roi crop. Besides, we convert them to support multi-image batch training.

* **For implementing in weakly supervised object detection, we change rpn to selectivesearch method firstly and change the latter part of fast rcnn to wsddn secondly.

## Benchmarking

We benchmark our code thoroughly on pascal voc 2007 datasets

