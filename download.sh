#!/bin/bash

TO_PATH=./msrn
python3 -c "import os; os.makedirs('$TO_PATH',exist_ok=True)"
wget https://image-quality-framework.s3-eu-west-1.amazonaws.com/iq-mfsr-use-case/models/weights/single_frame_sr/SISR_MSRN_X2_BICUBIC.pth -O $TO_PATH/model.pth

TO_PATH=.
wget https://image-quality-framework.s3-eu-west-1.amazonaws.com/iq-mfsr-use-case/datasets/xview-test-homo-v5-200samples.tar.gz -O $TO_PATH/xview-test-homo-v5-200samples.tar.gz
tar xvzf $TO_PATH/xview-test-homo-v5-200samples.tar.gz -C $TO_PATH
rm $TO_PATH/xview-test-homo-v5-200samples.tar.gz
