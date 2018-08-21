#code taken from https://github.com/meliketoy/wide-resnet.pytorch
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

import sys
import numpy as np

__all__ = ['teacher_wideresnet', 'student_wideresnet']
teacher_options = {'widen_factor':10, 'depth':34, 'dropout_rate':0.3, 'num_classes':100}
student_options = {'widen_factor':6, 'depth':22, 'dropout_rate':0.3, 'num_classes':100}

#TODO: Some of the things are not equal to the model definition (from the authors)
# which is here: https://github.com/szagoruyko/functional-zoo/blob/master/wide-resnet-50-2-export.ipynb

def conv3x3(in_planes, out_planes, stride=1):
    #TODO: Authors use, in their conv2d a padding=0 by default if I am not mistaken
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(self.topk_act(self.bn1(x))))
        out = self.conv2(self.topk_act(self.bn2(out)))
        out += self.shortcut(x)

        return out

class TopK_Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes,K=32):
        super(TopK_Wide_ResNet, self).__init__()
        self.in_planes = 16
		self.topk_act = topk_activation(K)

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = int((depth-4)/6)
        k = widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0]) #TODO: authors use stride=2, padding=3 in first convolution
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)
        self.apply(conv_init)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x) #TODO: after first layer they use relu and maxpool2d with parameters 3, 2, 1
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.topk_act(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out



def topk_wideresnet():
    model = TopK_Wide_ResNet(**student_options)
    return model


class topk_activation(nn.Module):

    """
    Top-k activation. It will keep the top-k values of the input and set to 0 everything else.
    Example:
        >>> a = Variable(torch.rand(4,4)*2-1, requires_grad=True)
        >>> a
        Variable containing:
        -0.0034 -0.5224  0.0061  0.3532
         0.9689 -0.9215  0.9913  0.6159
         0.2036  0.7713  0.1202  0.6936
         0.7343  0.4061  0.5276  0.7241
        [torch.FloatTensor of size 4x4]
        >>> topk_activation(5)(a)
        Variable containing:
         0.0000  0.0000  0.0000  0.0000
         0.9689 -0.9215  0.9913  0.0000
         0.0000  0.7713  0.0000  0.0000
         0.7343  0.0000  0.0000  0.0000
        [torch.FloatTensor of size 4x4]
        >>> topk_activation(5)(a).sum().backward()
        >>> a.grad
        Variable containing:
         0  0  0  0
         1  1  1  0
         0  1  0  0
         1  0  0  0
        [torch.FloatTensor of size 4x4]
    NOTE ABOUT THE BACKWARD PASS:
        This function is not differentiable on the entire real line, but it's almost everywhere differentiable.
        No special work is necessary in the backward pass and the simple gradient from the double_thresholding function
        is enough.
        In fact, let h(x, M) be the double thresholding function as above; h(x, M) = x if |x| >= M,
                                                                                     0 otherwise
        The gradient of this function is trivial to compute if M is constant; but if M (the value which we use for
        thresholding) depends on x (like in this case) then things are more complicated. The gradient of h(x, M) wrt to
        x can be written as
        d h(x, M)/dx = partial h(x, M)/partial x + partial h(x, M)/partial M * partial M/partial x.
        The first addend is the term where M is constant, so the gradient is the one coming from the double_thresholding function.
        partial M/partial x depends on the method we use to compute the value M given our input x, and need not be
        differentiable. But since partial h(x, M)/partial M is 0 almost everywhere (and undefined when x=M), then the
        term partial M/partial x doesn't matter and we are left with d h(x, M)/dx = partial h(x, M)/partial x,
        which is why no special work is required.
    """

    def __init__(self, K):
        super(topk_activation, self).__init__()
        self.K = K
        self.eps = 1e-10

    def forward(self, input):

        # We use K+1 and then add a very small value (self.eps) to the thresholding function
        # Using K directly could result in a situation where more than K non-zero values could be returned.
        # Instead we consider the value of the K+1-th biggest element, then we exclude it by adding eps
        # The result is that there will be at most K elements in the output vector (there could be less though)
        value = torch.topk(input.data.abs().view(-1), self.K + 1, dim=0)[0].min() + self.eps
        return double_threshold(value)(input)

    def extra_repr(self):
        str_ = "topk activation with K = {}".format(self.K)
        return str_
