import os
import shutil
import piq
import torch

from glob import glob
from scipy import ndimage
from typing import Any, Dict, Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt

from iquaflow.datasets import DSModifier, DSWrapper,DSModifier_jpg
from iquaflow.experiments import ExperimentInfo, ExperimentSetup
from iquaflow.experiments.experiment_visual import ExperimentVisual
from iquaflow.experiments.task_execution import PythonScriptTaskExecution
from iquaflow.metrics import BBDetectionMetrics, SNRMetric , RERMetric

from custom_iqf import DSModifierMFSR, SimilarityMetricsForMFSR, SlicedWassersteinMetric

#Define name of IQF experiment
experiment_name = "xview-test-v5-200samples"

#Define path of the original(reference) dataset
data_path = "./xviewds/xview-ds/test"

#DS wrapper is the class that encapsulate a dataset
ds_wrapper = DSWrapper(data_path=data_path)

#Define path of the training script
python_ml_script_path = 'custom_train.py'

#List of modifications that will be applied to the original dataset:

ds_modifiers_list = [
    DSModifierMFSR( params={
        'algo':algo,
        'zoom': 3,
        'n_jobs': 17
    } )
    for algo in [
        'fake',
        'warpw_v0',
        'warpw',
        'agk',
        'msrn'
        ]
]

ds_modifiers_list += [
    DSModifierMFSR( params={
        'algo':algo,
        'zoom': 3,
        'config':conf,
        'model':model
    } )
    for algo, conf, model in zip(
        [
            'hrn',
            'hrn',
            'hrn',
            'hrn',
            'hrn'
        ],
        [
            "hrn_exp27.json",
            "hrn_exp27-histmatch-v5.json",
            "hrn_exp27-histmatch-v6.json",
            "hrn_exp27+inria-histmatch-v5.json",
            "hrn_exp27+inria-histmatch-v6.json"
        ],
        [
            "exp27/HRNet_30.pth",
            "exp27-histmatch-v5/HRNet.pth",
            "exp27-histmatch-v6/HRNet.pth",
            "exp27+inria-histmatch-v5/HRNet.pth",
            "exp27+inria-histmatch-v6/HRNet.pth"
        ]
    )
]

# Task execution executes the training loop
# In this case the training loop is an empty script,
# this is because we evaluate the performance directly on the result of the modifiers.
task = PythonScriptTaskExecution( model_script_path = python_ml_script_path )

#Experiment definition, pass as arguments all the components defined beforehand
experiment = ExperimentSetup(
    experiment_name=experiment_name,
    task_instance=task,
    ref_dsw_train=ds_wrapper,
    ds_modifiers_list=ds_modifiers_list,
    repetitions=1
)

#Execute the experiment
experiment.execute()

# # ExperimentInfo is used to retrieve all the information of the whole experiment. 
# # It contains built in operations but also it can be used to retrieve raw data for futher analysis

experiment_info = ExperimentInfo(experiment_name)

print('Calculating similarity metrics...')

_ = experiment_info.apply_metric_per_run(
    SimilarityMetricsForMFSR( experiment_info, cut=12//2,  n_jobs=15 ),
    ds_wrapper.json_annotations,
)

print('Calculating Sliced Wasserstein Distance Metric...')

win = 128

_ = experiment_info.apply_metric_per_run(
    SlicedWassersteinMetric(
        experiment_info,
        n_jobs               = 15,
        ext                  = 'png',
        n_pyramids           = 1,
        slice_size           = 7,
        n_descriptors        = win*2,
        n_repeat_projection  = win,
        proj_per_repeat      = 4,
        device               = 'cpu',
        return_by_resolution = False,
        pyramid_batchsize    = win
    ),
    ds_wrapper.json_annotations,
)

print('Calculating RER Metric...')

_ = experiment_info.apply_metric_per_run(
    RERMetric(
        experiment_info,
        win=16,
        stride=16,
        ext="png",
        n_jobs=15
    ),
    ds_wrapper.json_annotations,
)

print('Calculating SNR Metric...')

_ = experiment_info.apply_metric_per_run(
    SNRMetric(
        experiment_info,
        ext="png",
        patch_sizes=[30],
        confidence_limit=50.0,
        n_jobs=15
    ),
    ds_wrapper.json_annotations,
)
