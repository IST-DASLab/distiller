python3.6 compress_classifier.py -a=alexnet --teacher_arch alexnet --teacher_weights /home/dan/.torch/models/alexnet-owt-4df8aa71.pth --lr=0.005 -p=50 ~/data-disk/ILSVRC2012/ --dataset imagenet -j 22 --epochs 90 --pretrained --compress=../agp-pruning/alexnet.schedule_agp_80.yaml --name="alexnet_80_conv_distilled"

#python3.6 compress_classifier.py -a=alexnet --lr=0.005 -p=50 ~/data-disk/ILSVRC2012/ --dataset imagenet -j 22 --epochs 90 --pretrained --compress=../agp-pruning/alexnet.schedule_agp.yaml --name="alexnet_80_conv_90_fc"


#python3.6 compress_classifier.py -a=alexnet --teacher_arch alexnet --teacher_weights /home/dan/.torch/models/alexnet-owt-4df8aa71.pth --lr=0.005 -p=50 ~/data-disk/ILSVRC2012/ --dataset imagenet -j 22 --epochs 90 --pretrained --compress=../agp-pruning/alexnet.schedule_agp.yaml --name="alexnet_80_conv_90_fc_distilled_init"


