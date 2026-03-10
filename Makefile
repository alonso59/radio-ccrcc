.PHONY: train_autoencoder_hydra_parallel

dev ?= 0
group ?= NG
fold ?= 0
task ?= autoencoder

train_seg_320_3d_fullres:
	CUDA_VISIBLE_DEVICES=$(dev) nnUNetv2_train 320 3d_fullres $(fold) -pretrained_weights data/nnUNet_results/Dataset220_KiTS2023/nnUNetTrainer__nnUNetResEncUNetPlans__3d_fullres/fold_0/checkpoint_best.pth -p nnUNetResEncUNetPlans --npz > nohup_$(fold)_320 2>&1 & echo "PID: $$!"

process_dicom_tcga:
	@echo "Processing DICOM files in TCGA dataset..."
	@python -m preprocess.dicom2nii > nohup_dcm.out 2>&1 & echo "PID: $$!"
	@echo "Processing started in the background. Logs are not being saved to a file."

process_svs_tcga:
	@echo "Processing SVS files in TCGA dataset..."
	@nohup python preprocess/svs2h5.py > nohup_svs.out 2>&1 & echo "PID: $$!"
	@echo "Processing started in the background. Logs are not being saved to a file."

convert_dicom_tcga:
	@echo "Converting DICOM to NIfTI for TCGA dataset..."
	python -m src.converter.convert -i data/tcga_dicom -o data/dataset/Dataset820/ -c data/filtered_vessel_evaluation.csv
	@echo "Conversion started in the background. Logs are not being saved to a file."

segment:
	CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 nnUNetv2_predict -i data/dataset/Dataset820/nifti -o data/dataset/Dataset820/nnunet/ -d 220 -c 3d_fullres -p nnUNetResEncUNetPlans -f 0 -chk checkpoint_best.pth > nohup_seg.out 2>&1 & echo "PID: $$!"

post_processing:
	nnUNetv2_apply_postprocessing -i data/dataset/Dataset820/nnunet/ -o data/dataset/Dataset820/seg/ -pp_pkl_file /home/alonso/Documents/radio-ccrcc/data/nnUNet_results/Dataset220_KiTS2023/nnUNetTrainer__nnUNetResEncUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/postprocessing.pkl -np 8 -plans_json /home/alonso/Documents/radio-ccrcc/data/nnUNet_results/Dataset220_KiTS2023/nnUNetTrainer__nnUNetResEncUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/plans.json

planner:
	python src/planner.py --config config/planner.yaml --splits

train_autoencoder:
	CUDA_VISIBLE_DEVICES=$(dev) python main.py dataset.fold=${fold} > nohup_${fold}.out 2>&1 & echo "PID: $$!"
