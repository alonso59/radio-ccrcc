fold ?= 0

train_autoencoder:
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python src/main.py dataset.fold=${fold} > nohup_${fold}.out 2>&1 & echo "PID: $$!"