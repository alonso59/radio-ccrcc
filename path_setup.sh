#!/usr/bin/env bash

mkdir -p data/nnUNet_raw_data_base
mkdir -p data/nnUNet_preprocessed
mkdir -p data/nnUNet_results

export nnUNet_raw=$PWD/data/nnUNet_raw_data_base
export nnUNet_preprocessed=$PWD/data/nnUNet_preprocessed
export nnUNet_results=$PWD/data/nnUNet_results

echo $nnUNet_raw
echo $nnUNet_preprocessed
echo $nnUNet_results