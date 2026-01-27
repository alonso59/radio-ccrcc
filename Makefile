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

segment:
	CUDA_VISIBLE_DEVICES=$(dev) nnUNetv2_predict -i data/dataset/tcga_kirc_nii/$(group) -o data/dataset/tcga_kirc_nnunet/$(group) -d 220 -c 3d_fullres -p nnUNetResEncUNetPlans -f 0 -chk checkpoint_best.pth

post_processing:
	nnUNetv2_apply_postprocessing -i data/dataset/tcga_kirc_nnunet/$(group) -o data/dataset/tcga_kirc_seg/$(group) -pp_pkl_file /home/alonso/Documents/radio-ccrcc/data/nnUNet_results/Dataset220_KiTS2023/nnUNetTrainer__nnUNetResEncUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/postprocessing.pkl -np 8 -plans_json /home/alonso/Documents/radio-ccrcc/data/nnUNet_results/Dataset220_KiTS2023/nnUNetTrainer__nnUNetResEncUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/plans.json
	
train_autoencoder:
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py dataset.fold=${fold} > nohup_${fold}.out 2>&1 & echo "PID: $$!"