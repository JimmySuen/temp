mkdir output_s1

python -m torch.distributed.launch --nproc_per_node=8 evaluation/semi_supervised/eval_cls.py --pretrained_weights /data/home/xias/projects/ibot/pretrained_models/ViTS16/checkpoint.pth --avgpool_patchtokens 0 --arch vit_small --checkpoint_key teacher --output_dir output_s1/ --finetune_head_layer 1 --epochs 1000 --lr 5e-6 --data_path /mnt/blobfuse/data/ImageNetS1/


mkdir output_s10

python -m torch.distributed.launch --nproc_per_node=8 evaluation/semi_supervised/eval_cls.py --pretrained_weights /data/home/xias/projects/ibot/pretrained_models/ViTS16/checkpoint.pth --avgpool_patchtokens 0 --arch vit_small --checkpoint_key teacher --output_dir output_s10/ --finetune_head_layer 1 --epochs 1000 --lr 5e-6 --data_path /mnt/blobfuse/data/ImageNetS10/


