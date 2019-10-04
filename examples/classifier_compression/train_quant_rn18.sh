#!/usr/bin/env bash
EXP_NAME="RN18_imagenet_train_quantized"
CHECKPOINT_ROOT="/home/dan/distiller-micronet/checkpoints"
ARCH="resnet18"
DATAPATH="/home/dan/ILSVRC2012"
#CHECKPOINT="/home/alex/distiller-MicroNet/imagenet_efficientnet/checkpoints/effnet_imagenet_prune_base2_best.pth.tar"

LR=0.005
MOM=0.9
WD=0.0

#python compress_classifier.py \
#	--name=$EXP_NAME \
#	--arch=$ARCH $DATAPATH \
#	--epochs=300 -p=30 -b=256 -j=8 \
#	--lr=$LR --momentum=$MOM --wd=$WD \
#	--pretrained \
#	--vs=0 \
#	--out-dir=$CHECKPOINT_ROOT	--gpus 0,1,2,3 \
#	--reset-optimizer \
#	--compress=./schedules/quantize_aware_training.yaml \
#	--kd-teacher=resnet34 \
#	--kd-pretrained \
#	--kd-temperature=5 \
#	--kd-student-wt=0.3 \
#	--kd-start-epoch=0 \
	
	
CUDA_VISIBLE_DEVICES=0,1,2,3 python compress_classifier.py \
	--name="RN18_QBASE"\
	--arch=$ARCH $DATAPATH \
	--epochs=10 -p=30 -b=256 -j=8 \
	--lr=$LR --momentum=$MOM --wd=$WD \
	--pretrained \
	--vs=0 \
	--out-dir=$CHECKPOINT_ROOT	\
	--compress=../quantization/preact_resnet18_imagenet_dorefa.yaml


CUDA_VISIBLE_DEVICES=0,1,2,3 python compress_classifier.py \
	--name="RN18_QKD_2"\
	--arch=$ARCH $DATAPATH \
	--epochs=10 -p=30 -b=256 -j=8 \
	--lr=$LR --momentum=$MOM --wd=$WD \
	--pretrained \
	--vs=0 \
	--out-dir=$CHECKPOINT_ROOT	\
	--compress=../quantization/preact_resnet18_imagenet_dorefa.yaml \
    --kd_type 2 --weight_distillation_loss=0.7 --teacher_arch resnet34 --temp_distillation 5


CUDA_VISIBLE_DEVICES=0,1,2,3 python compress_classifier.py \
	--name="RN18_QKD_BASE"\
	--arch=$ARCH $DATAPATH \
	--epochs=10 -p=30 -b=256 -j=8 \
	--lr=$LR --momentum=$MOM --wd=$WD \
	--pretrained \
	--vs=0 \
	--out-dir=$CHECKPOINT_ROOT	\
	--compress=../quantization/preact_resnet18_imagenet_dorefa.yaml \
    --kd_type 0 --weight_distillation_loss=0.7 --teacher_arch resnet34 --temp_distillation 5

CUDA_VISIBLE_DEVICES=0,1,2,3 python compress_classifier.py \
	--name="RN18_QKD_1"\
	--arch=$ARCH $DATAPATH \
	--epochs=10 -p=30 -b=256 -j=8 \
	--lr=$LR --momentum=$MOM --wd=$WD \
	--pretrained \
	--vs=0 \
	--out-dir=$CHECKPOINT_ROOT	\
	--compress=../quantization/preact_resnet18_imagenet_dorefa.yaml \
    --kd_type 1 --weight_distillation_loss=0.7 --teacher_arch resnet34 --temp_distillation 5