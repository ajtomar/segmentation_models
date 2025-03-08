
#!/bin/bash

export nnUNet_N_proc_DA=36
export nnUNet_codebase="path to codebase" # replace with your codebase
export nnUNet_raw_data_base="path to nnUNet_raw_data_base" # replace with your database
export nnUNet_preprocessed="path to nnUNet_preprocessed"
export RESULTS_FOLDER="path to result folder"
export ERROR_FOLDER="path to error folder"


CONFIG='path to config file'

echo "Config: ${CONFIG}"

### unit test
fold=0
echo "run on fold: ${fold}"
nnunet_use_progress_bar=1 CUDA_VISIBLE_DEVICES=0 \
        python3 -m torch.distributed.launch --master_port=4322 --nproc_per_node=1 \
        train.py --fold=${fold} --config=$CONFIG --resume='' --validation_only 

