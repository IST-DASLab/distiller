#train wide_resnet22 (student) from scratch 

#CUDA_VISIBLE_DEVICES=0,1 python3.6 compress_classifier.py --arch teacher_wideresnet ../../data.cifar100 --dataset=cifar100 -p=50 --lr=0.3 --epochs=200 -j=4 --compress=../agp-pruning/wideresnet22.schedule_0.yaml --name="teacher_resnet_cifar_100"
#CUDA_VISIBLE_DEVICES=2,3 python3.6 compress_classifier.py --arch student_wideresnet ../../data.cifar100 --dataset=cifar100 -p=50 --lr=0.3 --epochs=200 -j=4 --compress=../agp-pruning/wideresnet22.schedule_0.yaml --name="student_resnet_cifar_100"

#CUDA_VISIBLE_DEVICES=0,1 python3.6 compress_classifier.py --arch student_wideresnet ../../data.cifar100 --dataset=cifar100 -p=50 --lr=0.3 --epochs=350 -j=4 --compress=../agp-pruning/wideresnet22.schedule_95.yaml --name="teacher_resnet_cifar_100"

python3.6 compress_classifier.py --arch student_wideresnet --teacher_arch student_wideresnet --teacher_weights ./logs/student_resnet_cifar_100___2018.07.27-110549/student_resnet_cifar_100_best.pth.tar ../../data.cifar100 --dataset=cifar100 -p=50 --lr=0.3 --epochs=400 -j=8 --compress=../agp-pruning/wideresnet22.schedule_95.yaml --name="self_distilled_resnet_22_cifar_100_95_pruning"

python3.6 compress_classifier.py --arch student_wideresnet --teacher_arch student_wideresnet --teacher_weights ./logs/student_resnet_cifar_100___2018.07.27-110549/student_resnet_cifar_100_best.pth.tar ../../data.cifar100 --start_distillation_from_epoch=230 --dataset=cifar100 -p=50 --lr=0.3 --epochs=400 -j=8 --compress=../agp-pruning/wideresnet22.schedule_95.yaml --name="self_distilled_from_230_resnet_22_cifar_100_95_pruning"

python3.6 compress_classifier.py --arch student_wideresnet --teacher_arch student_wideresnet --teacher_weights ./logs/student_resnet_cifar_100___2018.07.27-110549/student_resnet_cifar_100_best.pth.tar ../../data.cifar100 --start_distillation_from_epoch=100 --dataset=cifar100 -p=50 --lr=0.3 --epochs=400 -j=8 --compress=../agp-pruning/wideresnet22.schedule_95.yaml --name="self_distilled_from_100_resnet_22_cifar_100_95_pruning"

python3.6 compress_classifier.py --arch student_wideresnet --teacher_arch student_wideresnet --teacher_weights ./logs/student_resnet_cifar_100___2018.07.27-110549/student_resnet_cifar_100_best.pth.tar ../../data.cifar100 --start_distillation_from_epoch=300 --dataset=cifar100 -p=50 --lr=0.3 --epochs=400 -j=8 --compress=../agp-pruning/wideresnet22.schedule_95.yaml --name="self_distilled_from_300_resnet_22_cifar_100_95_pruning"

python3.6 compress_classifier.py --arch student_wideresnet --teacher_arch teacher_wideresnet --teacher_weights ../../checkpoints/teacher_resnet_28_cifar_100_checkpoint.pth.tar ../../data.cifar100 --start_distillation_from_epoch=0 --dataset=cifar100 -p=50 --lr=0.3 --epochs=400 -j=8 --compress=../agp-pruning/wideresnet22.schedule_95.yaml --name="28x10_distilled_from_0_resnet_22_cifar_100_95_pruning"




#CUDA_VISIBLE_DEVICES=0 python3.6 compress_classifier.py --arch student_wideresnet ../../data.cifar100 --dataset=cifar100 -p=50 --lr=0.3 --epochs=200 -j=1 --name="student_resnet_22_cifar_100"

# train wide_resnet28 (teacher) from scratch

#CUDA_VISIBLE_DEVICES=0 python3.6 compress_classifier.py --arch teacher_wideresnet ../../data.cifar100 --dataset=cifar100 -p=50 --lr=0.3 --epochs=200 -j=1 --name="teacher_resnet_28_cifar_100"

# try out various pruning schedules for the student 

#python3.6 compress_classifier.py --arch student_wideresnet ../../data.cifar100 --dataset=cifar100 -p=50 --lr=0.3 --epochs=300 -j=1 --compress=../agp-pruning/wideresnet22.schedule_98.yaml --name="student_resnet_22_cifar_100_98percent_pruning"

#CUDA_VISIBLE_DEVICES=0 python3.6 compress_classifier.py --arch student_wideresnet --teacher_arch teacher_wideresnet --teacher_weights ./logs/teacher_resnet_28_cifar_100___2018.07.17-080637/teacher_resnet_28_cifar_100_checkpoint.pth.tar ../../data.cifar100 --dataset=cifar100 -p=50 --lr=0.3 --epochs=300 -j=1 --compress=../agp-pruning/wideresnet22.schedule_98.yaml --name="distilled_student_resnet_22_cifar_100_98percent_pruning"

#CUDA_VISIBLE_DEVICES=0 python3.6 compress_classifier.py --arch student_wideresnet ../../data.cifar100 --dataset=cifar100 -p=50 --lr=0.3 --epochs=300 -j=1 --compress=../agp-pruning/wideresnet22.schedule_95.yaml --name="student_resnet_22_cifar_100_95percent_pruning"

#python3.6 compress_classifier.py --arch student_wideresnet --teacher_arch teacher_wideresnet --teacher_weights ../../checkpoints/teacher_resnet_28_cifar_100_checkpoint.pth.tar ../../data.cifar100 --dataset=cifar100 -p=50 --lr=0.3 --epochs=300 -j=4 --compress=../agp-pruning/wideresnet22.schedule_98.yaml --name="distilled_student_resnet_22_cifar_100_98percent_pruning"

#python3.6 compress_classifier.py --arch student_wideresnet --teacher_arch teacher_wideresnet --teacher_weights ../../checkpoints/teacher_resnet_28_cifar_100_checkpoint.pth.tar ../../data.cifar100 --dataset=cifar100 -p=50 --lr=0.3 --epochs=300 -j=4 --compress=../agp-pruning/wideresnet22.schedule_95.yaml --name="distilled_student_resnet_22_cifar_100_95percent_pruning"

#python3.6 compress_classifier.py --arch student_wideresnet ../../data.cifar100 --dataset=cifar100 -p=50 --lr=0.3 --epochs=400 -j=4 --compress=../agp-pruning/wideresnet22.schedule_95.yaml --name="k80-student_resnet_22_cifar_100_95percent_pruning"

#python3.6 compress_classifier.py --arch student_wideresnet ../../data.cifar100 --dataset=cifar100 -p=50 --lr=0.3 --epochs=400 -j=4 --compress=../agp-pruning/wideresnet22.schedule_98.yaml --name="k80_student_resnet_22_cifar_100_98percent_pruning"

#python3.6 compress_classifier.py --arch student_wideresnet --teacher_arch teacher_wideresnet --teacher_weights ../../checkpoints/teacher_resnet_28_cifar_100_checkpoint.pth.tar ../../data.cifar100 --dataset=cifar100 -p=50 --lr=0.3 --epochs=200 -j=4 --compress=../ssl/resnet20_cifar_baseline_training.yaml --name="distilled_student_resnet_22_cifar_100_no_pruning"

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
