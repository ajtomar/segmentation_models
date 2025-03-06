# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch train_sam.py 

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=1 train_sam.py --config path_to_configs/cod-sam-vit-b.yaml 
