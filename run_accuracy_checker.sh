#!/bin/bash
export OMZ_DIR=./venv/lib/python3.8/site-packages/openvino/model_zoo
export DEFINITIONS_FILE=$OMZ_DIR/data/dataset_definitions.yml
export DATA_DIR=dataset
accuracy_check -c accuracy-check_wav2vec2-large-xlsr-53-japanese.yml -m public/wav2vec2-large-xlsr-53-japanese/FP32/wav2vec2-large-xlsr-53-japanese.xml -td CPU -tf openvino

