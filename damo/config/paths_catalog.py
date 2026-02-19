# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
"""Centralized catalog of paths."""
import os


class DatasetCatalog(object):
    DATA_DIR = 'datasets'
    DATASETS = {
        'coco_2017_train': {
            'img_dir': 'coco/train2017',
            'ann_file': 'coco/annotations/instances_train2017.json'
        },
        'coco_2017_val': {
            'img_dir': 'coco/val2017',
            'ann_file': 'coco/annotations/instances_val2017.json'
        },
        'coco_2017_test_dev': {
            'img_dir': 'coco/test2017',
            'ann_file': 'coco/annotations/image_info_test-dev2017.json'
        },
        # aliases used in da-damoyolo
        'coco_train': {
            'img_dir': 'coco/train/images',
            'ann_file': 'coco/train/annotations/train.json'
        },
        'coco_val': {
            'img_dir': 'coco/val/images',
            'ann_file': 'coco/val/annotations/val.json'
        },
        'coco_test': {
            'img_dir': 'coco/test/images',
            'ann_file': 'coco/test/annotations/test.json'
        },
        # 50-image calibration subset prepared in this workspace
        'coco_dropper_wire_calib50': {
            'img_dir': '/home/td/jszp-dev/damoyolo_quant/artifacts/data/dropper_wire_calib50_coco/images',
            'ann_file': '/home/td/jszp-dev/damoyolo_quant/artifacts/data/dropper_wire_calib50_coco/annotations/instances_dropper_wire_calib50.json'
        },
    }

    @staticmethod
    def get(name):
        if 'coco' in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs['img_dir']),
                ann_file=os.path.join(data_dir, attrs['ann_file']),
            )
            return dict(
                factory='COCODataset',
                args=args,
            )
        else:
            raise RuntimeError('Only support coco format now!')
        return None
