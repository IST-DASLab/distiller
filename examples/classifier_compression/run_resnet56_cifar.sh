#train resnet20 from scratch

CUDA_VISIBLE_DEVICES=0 python3.6 compress_classifier.py --arch resnet20_cifar  ../data.cifar10 -p=50 --lr=0.3 --epochs=180 --compress=../ssl/resnet20_cifar_baseline_training.yaml -j=1 --name="resnet20_cifar_baseline_run"