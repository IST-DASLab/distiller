python3.6 compress_classifier.py -a=alexnet --lr=0.005 -p=50 ~/data-disk/ILSVRC2012/ --dataset imagenet -j 22 --epochs 90 --pretrained --compress=../agp-pruning/alexnet.schedule_agp.yaml


python3.6 compress_classifier.py -a=alexnet --teacher_arch alexnet --teacher_weights /home/dan/.torch/models/alexnet-owt-4df8aa71.pth --lr=0.005 -p=50 ~/data-disk/ILSVRC2012/ --dataset imagenet -j 22 --epochs 90 --pretrained --compress=../agp-pruning/alexnet.schedule_agp.yaml


