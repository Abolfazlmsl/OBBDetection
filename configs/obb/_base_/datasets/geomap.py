dataset_type = 'DOTADataset'
data_root = '/home/DATA/RAMDISK/OBBDetection/mmdata/GeoMap/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

classes = (
    'Landslide1', 'Strike', 'Spring1', 'Minepit1', 'Hillside', 'Feuchte',
    'Torf', 'Bergsturz', 'Landslide2', 'Spring2', 'Spring3',
    'Minepit2', 'SpringB2', 'HillsideB2'
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadOBBAnnotations', with_bbox=True,
         with_label=True, obb_as_mask=True),
    dict(type='LoadDOTASpecialInfo'),
    dict(type='OBBRandomFlip', h_flip_ratio=0.0, v_flip_ratio=0.0),
    dict(
    type='RandomOBBRotate',
    rotate_after_flip=False,
    angles=(0,0),  
    vert_rate=0.0, 
    vert_cls=[]
    ),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DOTASpecialIgnore', ignore_size=2),
    dict(type='FliterEmpty'),
    dict(type='Mask2OBB', obb_type='obb'),
    dict(type='OBBDefaultFormatBundle'),
    dict(type='OBBCollect', keys=['img', 'gt_bboxes', 'gt_obboxes', 'gt_labels'])
]


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipRotateAug',
        img_scale=(128, 128),
        h_flip=False,
        v_flip=False,
        rotate=False,
        transforms=[
            dict(type='Resize', keep_ratio=True), 
            dict(type='OBBRandomFlip', h_flip_ratio=0.0, v_flip_ratio=0.0),
            dict(type='RandomOBBRotate', rotate_after_flip=False, angles=(0,0), vert_rate=0.0, vert_cls=[]), 
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='OBBCollect', keys=['img']),
        ])
]



data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type, 
        task='Task1',
        ann_file=data_root + 'train_split/annfiles/',
        img_prefix=data_root + 'train_split/images/',
        classes=classes,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type, 
        task='Task1',
        ann_file=data_root + 'val_split/annfiles/',
        img_prefix=data_root + 'val_split/images/',
        classes=classes,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        task='Task1',
        ann_file=data_root + 'test_split/annfiles/',
        img_prefix=data_root + 'test_split/images/',
        classes=classes,
        pipeline=test_pipeline)
)

data_loader = dict(
    persistent_workers=False,  
    prefetch_factor=2,
    pin_memory=True
)

opencv_num_threads = 0
omp_num_threads = 1

evaluation = dict(metric='mAP')