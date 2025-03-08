export nnUNet_N_proc_DA=36
export nnUNet_codebase="path to codebase of 3d-transunet"
export nnUNet_raw_data_base="path to nnUNet_raw_data_base" # replace with your database
export nnUNet_preprocessed="path to nnUNet_preprocessed"
export RESULTS_FOLDER="path to out folder"

CONFIG='path to config file'
export save_folder='path to save folder'

NUM_GPUS=1

inference() {
	fold=4
	gpu=0
	extra=${@:3}
	echo "Extra ${extra}"

	echo "inference: fold ${fold} on gpu ${gpu}, extra ${extra}"
	CUDA_VISIBLE_DEVICES=${gpu} \
		inference.py --config=${config} \
		--fold=${fold} --raw_data_folder ${subset} \
		--save_folder=${save_folder}/fold_${fold} ${extra} 
}

compute_metric() {
	fold=4
	pred_dir=${2:-${save_folder}/fold_${fold}/}
	extra=${@:3}

	echo "compute_metric: fold ${fold}, extra ${extra}, pred ${pred}"
	echo "raw_data_dir ${raw_data_dir}"
	python3 measure_dice.py \
		--config=${config} --fold=${fold} \
		--raw_data_dir=${raw_data_dir} \
		--pred_dir=${pred_dir} ${extra} 
}

fold=4
if [[ ${fold} == "all" ]]; then
	gpu=${2:-${gpu}}
else
	gpu=$((${fold} % ${NUM_GPUS}))
	gpu=${2:-${gpu}}
fi
extra=${@:3}
gpu=0
echo "extra: ${extra}"

# 5 fold eval
subset='imagesTr'

inference ${fold} ${gpu} ${extra}
# compute_metric ${fold}

echo "finished: inference: fold ${fold} on ${config}"
exit

# test set eval
# subset='imagesTs'
# inference ${fold} ${gpu} ${extra} --disable_split
# compute_metric ${fold} ${save_folder}/fold_${fold}/ --eval_mode Ts

# multi_save_folder=./results/inference/test/task008/encoderonly/fold_${fold},./results/inference/test/task008/decoderonly/fold_${fold}
# compute_metric ${fold} ${multi_save_folder} --eval_mode Ts
