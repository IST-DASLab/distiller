#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch
from .pruner import _ParameterPruner
import distiller
import random

def unmask_values_random(mask, number_to_unmask):

    '''
    :param mask: A 1d tensor consisting only of zeros and ones
    :param number_to_unmask: How many zeros to flip to one; will be chosen uniformly at random
    :return: The modified mask

    This function implements reservoir sampling to achieve uniform sampling in unmasking the zeros.
    '''

    if number_to_unmask <= 0: return mask

    indices_unmasked_values = []
    count_zero_seen = 0
    for idx, value in enumerate(mask):
        if value != 0: continue
        count_zero_seen += 1
        if len(indices_unmasked_values) < number_to_unmask:
            indices_unmasked_values.append(idx)
            continue
        if random.random() <= number_to_unmask/count_zero_seen:
            #keep new item and replace old one
            rand_idx = random.randint(0, number_to_unmask-1) #is this reliable for big number_to_unmask?
            indices_unmasked_values[rand_idx] = idx
        else: pass #keep old items, discard new one

    for idx in indices_unmasked_values:
        mask[idx] = 1

    return mask

class SparsityLevelParameterPruner(_ParameterPruner):
    """Prune to an exact pruning level specification.

    This pruner is very similar to MagnitudeParameterPruner, but instead of
    specifying an absolute threshold for pruning, you specify a target sparsity
    level (expressed as a fraction: 0.5 means 50% sparsity.)

    To find the correct threshold, we view the tensor as one large 1D vector, sort
    it using the absolute values of the elements, and then take topk elements.
    """

    def __init__(self, name, levels, **kwargs):
        super(SparsityLevelParameterPruner, self).__init__(name)
        self.levels = levels
        assert self.levels

    def set_param_mask(self, param, param_name, zeros_mask_dict, meta):
        # If there is a specific sparsity level specified for this module, then
        # use it.  Otherwise try to use the default level ('*').
        desired_sparsity = self.levels.get(param_name, self.levels.get('*', 0))
        if desired_sparsity == 0:
            return

        self.prune_level(param, param_name, zeros_mask_dict, desired_sparsity)

    @staticmethod
    def prune_level(param, param_name, zeros_mask_dict, desired_sparsity):
        expected_number_pruned_values = int(desired_sparsity * param.numel())
        #No need to sort, just take min later - it's O(k) vs O(k log k)
        bottomk, _ = torch.topk(param.data.abs().view(-1), expected_number_pruned_values, largest=False, sorted=True)
        threshold = bottomk.data[-1] # This is the largest element from the group of elements that we prune away
        parameter_weight_mask = distiller.threshold_mask(param.data, threshold)

        # The percentage of pruned weights now may be larger or equal than the desired sparsity.
        # We would like to make sure that it's actually equal, otherwise it will enforce a lot more weights to be zero than expected
        number_pruned_values = int((parameter_weight_mask==0).int().sum())

        if number_pruned_values < expected_number_pruned_values:
            raise ValueError('Number of pruned values should always be higher than the desired sparsity')

        # We will unmask values at random so to ensure that the final mask has the desired sparsity
        number_values_to_unmask = number_pruned_values-expected_number_pruned_values
        parameter_weight_mask = unmask_values_random(parameter_weight_mask.view(-1), number_values_to_unmask)
        zeros_mask_dict[param_name].mask = parameter_weight_mask.view_as(param.data)

        # some debug information - can delete after code works correctly
        print('Expected percentage of pruned values: {}'.format(desired_sparsity))
        print('Actual percentage before fix: {}'.format(number_pruned_values / param.numel()))
        print('Actual percentage after fix: {}'.format(int((parameter_weight_mask==0).int().sum()) / param.numel()))
