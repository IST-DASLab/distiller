#train resnet20 from scratch -> done resnet20_cifar_baseline_run___2018.07.08-064414

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

CUDA_VISIBLE_DEVICES=0 python3.6 compress_classifier.py --arch student_wideresnet ../../data.cifar100 --dataset=cifar100 -p=50 --lr=0.3 --epochs=600 --compress=../agp-pruning/resnet20.schedule_agp_progressive.yaml -j=1 --name="resnet_20_cifar_100"