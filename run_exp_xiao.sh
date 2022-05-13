mkdir outputs1_v10_base_mask08_1crops_droppath01_wd01

python -m torch.distributed.launch --nproc_per_node=8 evaluation/semi_supervised/eval_cls.py --pretrained_weights /mnt/blobfuse/projects/ibot/pretrained_models/zhirong/v10_base_mask08_1crops_droppath01_wd01/current.pth.tar --avgpool_patchtokens 0 --arch vit_base --checkpoint_key teacher --output_dir outputs1_v10_base_mask08_1crops_droppath01_wd01/ --finetune_head_layer 1 --epochs 1000 --lr 5e-6 --data_path /mnt/blobfuse/data/ImageNetS1/


mkdir outputs10_v10_base_mask08_1crops_droppath01_wd01

python -m torch.distributed.launch --nproc_per_node=8 evaluation/semi_supervised/eval_cls.py --pretrained_weights /mnt/blobfuse/projects/ibot/pretrained_models/zhirong/v10_base_mask08_1crops_droppath01_wd01/current.pth.tar --avgpool_patchtokens 0 --arch vit_base --checkpoint_key teacher --output_dir outputs10_v10_base_mask08_1crops_droppath01_wd01/ --finetune_head_layer 1 --epochs 1000 --lr 5e-6 --data_path /mnt/blobfuse/data/ImageNetS10/


