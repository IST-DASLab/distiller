#train wide_resnet22 (student) from scratch 

#CUDA_VISIBLE_DEVICES=0 python3.6 compress_classifier.py --arch student_wideresnet ../../data.cifar100 --dataset=cifar100 -p=50 --lr=0.3 --epochs=200 -j=1 --name="student_resnet_22_cifar_100"

# train wide_resnet28 (teacher) from scratch

#CUDA_VISIBLE_DEVICES=0 python3.6 compress_classifier.py --arch teacher_wideresnet ../../data.cifar100 --dataset=cifar100 -p=50 --lr=0.3 --epochs=200 -j=1 --name="teacher_resnet_28_cifar_100"

# try out various pruning schedules for the student 

#train original student

#CUDA_VISIBLE_DEVICES=0 python3.6 compress_classifier.py --arch student_wideresnet ../../data.cifar100 --dataset=cifar100 -p=50 --lr=0.3 --epochs=200 -j=1 --compress=../agp-pruning/wideresnet22.schedule_98.yaml --name="student_resnet_22_cifar_100_baseline"

#train original teacher

#CUDA_VISIBLE_DEVICES=0 python3.6 compress_classifier.py --arch teacher_wideresnet ../../data.cifar100 --dataset=cifar100 -p=50 --lr=0.3 --epochs=200 -j=1 --compress=../agp-pruning/wideresnet22.schedule_98.yaml --name="student_resnet_22_cifar_100_baseline"

CUDA_VISIBLE_DEVICES=0 python3.6 compress_classifier.py --arch student_wideresnet --act-stats  ../../data.cifar100 --dataset cifar100 -p=50 --lr=0.3 --epochs=300 --compress=../agp-pruning/wideresnet22.schedule_90.yaml -j=1 --deterministic --name="baseline_experiment"

CUDA_VISIBLE_DEVICES=0 python3.6 compress_classifier.py --arch student_wideresnet --act-stats ../../data.cifar100 --dataset=cifar100 -p=50 --lr=0.03 --epochs=100 -j=1 --compress=../agp-pruning/wideresnet22.pretrained.schedule_98.yaml --name="student_resnet_22_cifar_100_98percent_pruning"

CUDA_VISIBLE_DEVICES=0 python3.6 compress_classifier.py --arch student_wideresnet --teacher_arch student_wideresnet --teacher_weights ../../checkpoints/student_wideresnet_cifar_100.pth.tar ../../data.cifar100 --dataset=cifar100 -p=50 --lr=0.03 --epochs=100 -j=1 --compress=../agp-pruning/wideresnet22.pretrained.schedule_98.yaml --name="student_resnet_22_cifar_100_98percent_pruning_selfdistilled"

CUDA_VISIBLE_DEVICES=0 python3.6 compress_classifier.py --arch student_wideresnet --teacher_arch student_wideresnet --teacher_weights ../../checkpoints/student_wideresnet_cifar_100.pth.tar --distillation_weight 0.9 ../../data.cifar100 --dataset=cifar100 -p=50 --lr=0.03 --epochs=100 -j=1 --compress=../agp-pruning/wideresnet22.pretrained.schedule_98.yaml --name="student_resnet_22_cifar_100_98percent_pruning_selfdistilled_weight_0.9"

CUDA_VISIBLE_DEVICES=0 python3.6 compress_classifier.py --arch student_wideresnet --teacher_arch teacher_wideresnet --teacher_weights ../../checkpoints/teacher_wideresnet_cifar_100.pth.tar ../../data.cifar100 --dataset=cifar100 -p=50 --lr=0.03 --epochs=100 -j=1 --compress=../agp-pruning/wideresnet22.pretrained.schedule_98.yaml --name="student_resnet_22_cifar_100_98percent_pruning_teacherdistilled"

CUDA_VISIBLE_DEVICES=0 python3.6 compress_classifier.py --arch student_wideresnet --teacher_arch teacher_wideresnet --teacher_weights ../../checkpoints/teacher_wideresnet_cifar_100.pth.tar --start_distillation_from_epoch 20 ../../data.cifar100 --dataset=cifar100 -p=50 --lr=0.03 --epochs=100 -j=1 --compress=../agp-pruning/wideresnet22.pretrained.schedule_98.yaml --name="student_resnet_22_cifar_100_98percent_pruning_teacherdistilled_start20"





#CUDA_VISIBLE_DEVICES=0 python3.6 compress_classifier.py --arch student_wideresnet --teacher_arch teacher_wideresnet --teacher_weights ./logs/teacher_resnet_28_cifar_100___2018.07.17-080637/teacher_resnet_28_cifar_100_checkpoint.pth.tar ../../data.cifar100 --dataset=cifar100 -p=50 --lr=0.3 --epochs=300 -j=1 --compress=../agp-pruning/wideresnet22.schedule_98.yaml --name="distilled_student_resnet_22_cifar_100_98percent_pruning"

#CUDA_VISIBLE_DEVICES=0 python3.6 compress_classifier.py --arch student_wideresnet ../../data.cifar100 --dataset=cifar100 -p=50 --lr=0.3 --epochs=300 -j=1 --compress=../agp-pruning/wideresnet22.schedule_95.yaml --name="student_resnet_22_cifar_100_95percent_pruning"

#CUDA_VISIBLE_DEVICES=0 python3.6 compress_classifier.py --arch student_wideresnet --teacher_arch teacher_wideresnet --teacher_weights ./logs/teacher_resnet_28_cifar_100___2018.07.17-080637/teacher_resnet_28_cifar_100_checkpoint.pth.tar ../../data.cifar100 --dataset=cifar100 -p=50 --lr=0.3 --epochs=300 -j=1 --compress=../agp-pruning/wideresnet22.schedule_95.yaml --name="distilled_student_resnet_22_cifar_100_95percent_pruning"

#CUDA_VISIBLE_DEVICES=0 python3.6 compress_classifier.py --arch student_wideresnet --teacher_arch teacher_wideresnet --teacher_weights ./logs/teacher_resnet_28_cifar_100___2018.07.17-080637/teacher_resnet_28_cifar_100_checkpoint.pth.tar ../../data.cifar100 --dataset=cifar100 -p=50 --lr=0.3 --epochs=300 -j=1 --compress=../agp-pruning/wideresnet22.schedule_98.yaml --name="distilled_student_resnet_22_cifar_100_98percent_pruning"

#CUDA_VISIBLE_DEVICES=0 python3.6 compress_classifier.py --arch student_wideresnet --teacher_arch teacher_wideresnet --teacher_weights ./logs/teacher_resnet_28_cifar_100___2018.07.17-080637/teacher_resnet_28_cifar_100_best.pth.tar ../../data.cifar100 --dataset=cifar100 -p=50 --lr=0.3 --epochs=400 -j=1 --compress=../agp-pruning/wideresnet22.schedule_70.yaml --name="distilled_student_resnet_22_cifar_100_70percent_pruning"

#CUDA_VISIBLE_DEVICES=0 python3.6 compress_classifier.py --arch resnet20_cifar  ../data.cifar10 -p=50 --lr=0.3 --epochs=180 --compress=../ssl/resnet20_cifar_baseline_training.yaml -j=1 --name="resnet20_cifar_baseline_run"

#just distill resnet20 from 56
#CUDA_VISIBLE_DEVICES=0 python3.6 compress_classifier.py --teacher_arch resnet56_cifar --teacher_weights ./logs/resnet56_cifar_baseline_run___2018.07.08-072317/resnet56_cifar_baseline_run_best.pth.tar --start_distillation_from_epoch=0 --weight_distillation_loss=0.7 --arch resnet20_cifar ../data.cifar10 -p=50 --lr=0.3 --epochs=200 --compress=../ssl/resnet20_cifar_baseline_training.yaml -j=1 --name="resnet20_cifar_distilled_from_56"

#train pruned version of resnet20
#CUDA_VISIBLE_DEVICES=0 python3.6 compress_classifier.py --arch resnet20_cifar ../data.cifar10 -p=50 --lr=0.3 --epochs=300 --compress=../agp-pruning/resnet20.schedule_agp_aggressive.yaml -j=1 --name="new_resnet20_cifar_pruned_aggressive"

#perform aggressive pruning and distill from resnet_56
#CUDA_VISIBLE_DEVICES=0 python3.6 compress_classifier.py --teacher_arch resnet56_cifar --teacher_weights ./logs/resnet56_cifar_baseline_run___2018.07.08-072317/resnet56_cifar_baseline_run_best.pth.tar --start_distillation_from_epoch=0 --weight_distillation_loss=0.7 --arch resnet20_cifar ../data.cifar10 -p=50 --lr=0.3 --epochs=600 --compress=../agp-pruning/resnet20.schedule_agp_progressive.yaml -j=1 --name="progressive_pruner_resnet20_cifar_distilled_from_56"

#Goal: compare distilled pruning with the baselines
#We take Resnet20 as the target architecture




#CUDA_VISIBLE_DEVICES=0 python3.6 compress_classifier.py --arch resnet20_cifar ../../data.cifar100 -p=50 --lr=0.3 --epochs=600 --compress=../agp-pruning/resnet20.schedule_agp_progressive.yaml -j=1 --name="resnet_20_cifar_100"

#CUDA_VISIBLE_DEVICES=0 python3.6 compress_classifier.py --arch student_wideresnet ../../data.cifar100 --dataset=cifar100 ../../data.cifar100 -p=50 --lr=0.3 --epochs=600 --compress=../agp-pruning/resnet20.schedule_agp_progressive.yaml -j=1 --name="resnet_20_cifar_100"

#CUDA_VISIBLE_DEVICES=0 python3.6 compress_classifier.py --arch student_wideresnet ../../data.cifar100 --dataset=cifar100 -p=50 --lr=0.3 --epochs=600 --compress=../agp-pruning/resnet20.schedule_agp_progressive.yaml -j=1 --name="resnet_20_cifar_100"
