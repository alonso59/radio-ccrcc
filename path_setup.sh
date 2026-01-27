#!/usr/bin/env bash

mkdir -p data/nnUNet_raw_data_base
mkdir -p data/nnUNet_preprocessed
mkdir -p data/nnUNet_trained_models
mkdir -p data/nnUNet_results
mkdir -p data/nnUNet_seg_output_3d

export nnUNet_raw=$PWD/data/nnUNet_raw_data_base
export nnUNet_preprocessed=$PWD/data/nnUNet_preprocessed
export nnUNet_trained_models=$PWD/data/nnUNet_trained_models
export nnUNet_results=$PWD/data/nnUNet_results
export OUTPUT_DIRECTORY_3D=$PWD/data/nnUNet_seg_output_3d

echo $nnUNet_raw
echo $nnUNet_preprocessed
echo $nnUNet_trained_models
echo $nnUNet_results
echo $OUTPUT_DIRECTORY_3D
