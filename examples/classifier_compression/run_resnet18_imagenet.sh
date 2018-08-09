#run baseline for 75%
python3.6 compress_classifier.py -a=resnet18 --lr=0.001 -p=50 ~/data-disk/ILSVRC2012/ --dataset imagenet -j 22 --epochs 100 --pretrained --compress=../agp-pruning/resnet18.schedule_agp_75.yaml --name="resnet18_75_conv_baseline"

#run distilled for 75%
python3.6 compress_classifier.py -a=resnet18 --teacher_arch resnet18 --teacher_weights /home/dan/.torch/models/resnet18-5c106cde.pth --lr=0.001 -p=50 ~/data-disk/ILSVRC2012/ --dataset imagenet -j 22 --epochs 100 --pretrained --compress=../agp-pruning/resnet18.schedule_agp_75.yaml --name="resnet18_75_conv_self_distilled"

#run distilled for 75%
python3.6 compress_classifier.py -a=resnet18 --teacher_arch resnet152 --teacher_weights /home/dan/.torch/models/resnet152-b121ed2d.pth --lr=0.001 -p=50 ~/data-disk/ILSVRC2012/ --dataset imagenet -j 22 --epochs 100 --pretrained --compress=../agp-pruning/resnet18.schedule_agp_75.yaml --name="resnet18_75_conv_distilled_152"
