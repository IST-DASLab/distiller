#!/usr/bin/env bash
#run the original
#python3.6 compress_classifier.py --arch simplenet_cifar ../../../data.cifar10 -p 10 -j=1 --lr=0.01 --epochs=80 --name="original_simplenet"

#run the pruned original
#python3.6 compress_classifier.py --arch simplenet_cifar ../../../data.cifar10 -p 10 -j=1 --lr=0.01 --epochs=80 --compress=../agp-pruning/simplenet.aggressive.agp.yaml --name="pruned_simplenet"

#run the distilled original, with no pruning

python3.6 compress_classifier.py --teacher_weights ./logs/resnet20_cifar_baseline_run___2018.07.08-064414/resnet20_cifar_baseline_run_best.pth.tar --start_distillation_from_epoch=0 --weight_distillation_loss=0.7 --arch simplenet_cifar ../../../data.cifar10 -p 10 -j=1 --lr=0.01 --epochs=80 --kd_type 1 --name="distilled_simplenet_type_1"

python3.6 compress_classifier.py --teacher_weights ./logs/resnet20_cifar_baseline_run___2018.07.08-064414/resnet20_cifar_baseline_run_best.pth.tar --start_distillation_from_epoch=0 --weight_distillation_loss=0.7 --arch simplenet_cifar ../../../data.cifar10 -p 10 -j=1 --lr=0.01 --epochs=80 --kd_type 2 --name="distilled_simplenet_type_2"

python3.6 compress_classifier.py --teacher_weights ./logs/resnet20_cifar_baseline_run___2018.07.08-064414/resnet20_cifar_baseline_run_best.pth.tar --start_distillation_from_epoch=0 --weight_distillation_loss=0.7 --arch simplenet_cifar ../../../data.cifar10 -p 10 -j=1 --lr=0.01 --epochs=80 --name="distilled_simplenet_type_0"

#python3.6 compress_classifier.py --teacher_weights ./logs/resnet20_cifar_baseline_run___2018.07.08-064414/resnet20_cifar_baseline_run_best.pth.tar --start_distillation_from_epoch=0 --weight_distillation_loss=0.7 --arch simplenet_cifar ../../../data.cifar10 -p 10 -j=1 --lr=0.01 --epochs=80 --kd_type 1 --name="distilled_simplenet_type_1"



#python3.6 compress_classifier.py --arch simplenet_cifar --resume=./logs/2018.07.06-083116/checkpoint.pth.tar ../../../data.cifar10 -p 10 -j=1 --lr=0.01 --epochs=100 --compress=../agp-pruning/simplenet.agp.yaml --name="resume_simple"

#run the pruned distilled
#python3.6 compress_classifier.py --teacher_arch simplenet_cifar --teacher_weights ./logs/2018.07.06-083116/checkpoint.pth.tar --weight_distillation_loss=0.7 --arch simplenet_cifar --resume=./logs/2018.07.06-083116/checkpoint.pth.tar ../../../data.cifar10 -p 10 -j=1 --lr=0.01 --epochs=100 --compress=../agp-pruning/simplenet.agp.yaml --name="resume_distilled_weight_0.7"

#python3.6 compress_classifier.py --arch simplenet_cifar ../../../data.cifar10 -p 10 -j=1 --lr=0.01 --epochs=180 --compress=../agp-pruning/simplenet.aggressive.agp.yaml --name="simplenet_pruning_no_distill_aggressive_schedule"

#python3.6 compress_classifier.py --teacher_arch resnet20_cifar --teacher_weights ./logs/resnet20_cifar_baseline_run___2018.07.08-064414/resnet20_cifar_baseline_run_best.pth.tar --start_distillation_from_epoch=0 --weight_distillation_loss=0.7 --arch simplenet_cifar ../../../data.cifar10 -p 10 -j=1 --lr=0.01 --epochs=180 --compress=../agp-pruning/simplenet.aggressive.agp.yaml --name="pruning_distilled_simplenet_from_resnet20_superaggressive_from_epoch_0"

#python3.6 compress_classifier.py --teacher_arch simplenet_cifar --teacher_weights ./logs/2018.07.06-083116/checkpoint.pth.tar --weight_distillation_loss=0.5 --temp_distillation=5 --arch simplenet_cifar --resume=./logs/2018.07.06-083116/checkpoint.pth.tar ../../../data.cifar10 -p 10 -j=1 --lr=0.01 --epochs=100 --compress=../agp-pruning/simplenet.agp.yaml --name="resume_distilled_temp_5"


#python3.6 compress_classifier.py --arch basicnet_cifar ../../data.cifar10 -p 10 -j=1 --lr=0.01 --epochs=10 --name="basicnet"