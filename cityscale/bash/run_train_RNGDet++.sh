CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 main_train.py --savedir RNGDet\
 --dataroot ./dataset/ --batch_size 20 --ROI_SIZE 128 --nepochs 50 --backbone resnet101 --eos_coef 0.2\
 --lr 9e-5 --lr_backbone 9e-5 --weight_decay 1e-5 --noise 8 --image_size 2048\
  --candidate_filter_threshold 30 --logit_threshold 0.75 --extract_candidate_threshold 0.55 --alignment_distance 5\
  --instance_seg --multi_scale --current_best_model /home/m/Documents/RNGDetPlusPlus_1/cityscale/RNGDet_multi_ins/checkpoints/RNGDetPP_best.pt\
  --resume /home/m/Documents/RNGDetPlusPlus_1/cityscale/RNGDet_multi_ins/checkpoints/RNGDetPP_best.pt
