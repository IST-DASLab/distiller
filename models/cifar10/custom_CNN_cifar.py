import torch.nn as nn
import torch.nn.functional as Functional
from torch.nn.init import xavier_uniform, calculate_gain
import copy



#These are the paper specification for teacher model and student model
#Teacher model, 5.3 million parameters
teacherModelSpec = {'spec_conv_layers': [(76, 3, 3), (76, 3, 3), (126, 3, 3), (126, 3, 3), (148, 3, 3),
                                        (148, 3, 3), (148, 3, 3), (148, 3, 3)],
                    'spec_max_pooling': [(1,2,2), (3, 2, 2), (7, 2, 2)],
                    'spec_dropout_rates': [(1, 0.2), (3, 0.3), (7, 0.35), (8, 0.4), (9, 0.4)],
                    'spec_linear': [1200, 1200], 'width': 32, 'height': 32}

#smaller model to use as baseline, around 1 million parameters
smallerModelSpec = {'spec_conv_layers': [(75, 5, 5), (50, 5, 5), (50, 5, 5), (25, 5, 5)],
                    'spec_max_pooling': [(1, 2, 2), (3, 2, 2)],
                    'spec_dropout_rates': [(1, 0.2), (3, 0.3), (4, 0.4)],
                    'spec_linear': [500], 'width': 32, 'height': 32}

distilledModelSpec = copy.deepcopy(smallerModelSpec)
distilledModelSpec['spec_dropout_rates'] = []

__all__ = ['custom_cnn_teacher', 'custom_cnn_smaller', 'custom_cnn_student']


class ConvolForwardNet(nn.Module):

    ''' Teacher model as described in the paper :
    "Do deep convolutional neural network really need to be deep and convolutional?"'''

    def __init__(self, width, height, spec_conv_layers, spec_max_pooling, spec_linear, spec_dropout_rates, useBatchNorm=False,
                 useAffineTransformInBatchNorm=False, num_channels_input=3, num_outputs=10):

        '''
        The structure of the network is: a number of convolutional layers, intermittend max-pooling and dropout layers,
        and a number of linear layers. The max-pooling layers are inserted in the positions specified, as do the dropout
        layers.

        :param spec_conv_layers: list of tuples with (numFilters, width, height) (one tuple for each layer);
        :param spec_max_pooling: list of tuples with (posToInsert, width, height) of max-pooling layers
        :param spec_dropout_rates list of tuples with (posToInsert, rate of dropout) (applied after max-pooling)
        :param spec_linear: list with numNeurons for each layer (i.e. [100, 200, 300] creates 3 layers)
        '''


        super(ConvolForwardNet, self).__init__()


        self.width = width
        self.height = height
        self.conv_layers = []
        self.max_pooling_layers = []
        self.dropout_layers = []
        self.linear_layers = []
        self.max_pooling_positions = []
        self.dropout_positions = []
        self.useBatchNorm = useBatchNorm
        self.batchNormalizationLayers = []

        #creating the convolutional layers
        oldNumChannels = num_channels_input
        for idx in range(len(spec_conv_layers)):
            currSpecLayer = spec_conv_layers[idx]
            numFilters = currSpecLayer[0]
            kernel_size = (currSpecLayer[1], currSpecLayer[2])
            #The padding needs to be such that width and height of the image are unchanges after each conv layer
            padding = ((kernel_size[0]-1)//2, (kernel_size[1]-1)//2)
            newConvLayer = nn.Conv2d(in_channels=oldNumChannels, out_channels=numFilters,
                                                                    kernel_size=kernel_size, padding=padding)
            xavier_uniform(newConvLayer.weight, calculate_gain('conv2d')) #glorot weight initialization
            self.conv_layers.append(newConvLayer)
            self.batchNormalizationLayers.append(nn.BatchNorm2d(numFilters,
                                                            affine=useAffineTransformInBatchNorm))
            oldNumChannels = numFilters

        #creating the max pooling layers
        for idx in range(len(spec_max_pooling)):
            currSpecLayer = spec_max_pooling[idx]
            kernel_size = (currSpecLayer[1], currSpecLayer[2])
            self.max_pooling_layers.append(nn.MaxPool2d(kernel_size))
            self.max_pooling_positions.append(currSpecLayer[0])

        #creating the dropout layers
        for idx in range(len(spec_dropout_rates)):
            currSpecLayer = spec_dropout_rates[idx]
            rate = currSpecLayer[1]
            currPosition = currSpecLayer[0]
            if currPosition < len(self.conv_layers):
                #we use dropout2d only for the conv_layers, otherwise we use the usual dropout
                self.dropout_layers.append(nn.Dropout2d(rate))
            else:
                self.dropout_layers.append(nn.Dropout(rate))
            self.dropout_positions.append(currPosition)


        #creating the linear layers
        oldInputFeatures = oldNumChannels * width * height // 2**(2*len(self.max_pooling_layers))
        for idx in range(len(spec_linear)):
            currNumFeatures = spec_linear[idx]
            newLinearLayer = nn.Linear(in_features=oldInputFeatures, out_features=currNumFeatures)
            xavier_uniform(newLinearLayer.weight, calculate_gain('linear'))  # glorot weight initialization
            self.linear_layers.append(newLinearLayer)
            self.batchNormalizationLayers.append(nn.BatchNorm1d(currNumFeatures,
                                                                                 affine=useAffineTransformInBatchNorm))
            oldInputFeatures = currNumFeatures

        #final output layer
        self.out_layer = nn.Linear(in_features=oldInputFeatures, out_features=num_outputs)
        xavier_uniform(self.out_layer.weight, calculate_gain('linear'))


        self.conv_layers = nn.ModuleList(self.conv_layers)
        self.max_pooling_layers = nn.ModuleList(self.max_pooling_layers)
        self.dropout_layers = nn.ModuleList(self.dropout_layers)
        self.linear_layers = nn.ModuleList(self.linear_layers)
        self.batchNormalizationLayers = nn.ModuleList(self.batchNormalizationLayers)
        self.num_conv_layers = len(self.conv_layers)
        self.total_num_layers = self.num_conv_layers + len(self.linear_layers)

    def forward(self, input):

        for idx in range(self.total_num_layers):
            if idx < self.num_conv_layers:
                input = Functional.relu(self.conv_layers[idx](input))
            else:
                if idx == self.num_conv_layers:
                    #if it is the first layer after the convolutional layers, make it as a vector
                    input = input.view(input.size()[0], -1)
                input = Functional.relu(self.linear_layers[idx-self.num_conv_layers](input))

            if self.useBatchNorm:
                input = self.batchNormalizationLayers[idx](input)

            try:
                posMaxLayer = self.max_pooling_positions.index(idx)
                input = self.max_pooling_layers[posMaxLayer](input)
            except ValueError: pass

            try:
                posDropoutLayer = self.dropout_positions.index(idx)
                input = self.dropout_layers[posDropoutLayer](input)
            except ValueError: pass

        input = Functional.relu(self.out_layer(input))

        #No need to take softmax if the loss function is cross entropy
        return input



def custom_cnn_teacher():
    model = ConvolForwardNet(**teacherModelSpec, useBatchNorm=True, useAffineTransformInBatchNorm=True)
    return model

def custom_cnn_smaller():
    model = ConvolForwardNet(**smallerModelSpec, useBatchNorm=True, useAffineTransformInBatchNorm=True)
    return model

def custom_cnn_student():
    model = ConvolForwardNet(**distilledModelSpec, useBatchNorm=True, useAffineTransformInBatchNorm=True)
    return model



