# coding=utf-8
# Copyright 2019 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wrapper around FeatureDict to allow better control over decoding.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_datasets.core.features import feature as feature_lib
from tensorflow_datasets.core.features import sequence_feature as sequence_lib


class _FeaturesManager(object):
  """Wrapper around the top-level `FeatureConnector` to manage decoding.

  The FeatureManager allow to have better control over the decoding, and
  eventually apply augmentations.

  The FeatureManager behave exactly the same as the top-level feature, but
  uses a custom decoding function.

  """

  def decode_example(self, serialized_example):
    """Decode the serialize examples.

    Args:
      serialized_example: Nested `dict` of `tf.Tensor`

    Returns:
      example: Nested `dict` containing the decoded nested examples.
    """

    # Step 1: Flatten the nested dict => []
    flat_example = self._flatten(serialized_example)
    flat_features = self._flatten(self)
    flat_serialized_info = self._flatten(self.get_serialized_info())
    flat_tensor_info = self._flatten(self.get_tensor_info())

    # Step 2: Apply the decoding
    flatten_decoded = []
    for feature, example, tensor_info, serialized_info in zip(
        flat_features, flat_example, flat_tensor_info, flat_serialized_info):
      flatten_decoded.append(_decode_feature(
          feature=feature,
          example=example,
          tensor_info=tensor_info,
          serialized_info=serialized_info,
      ))

    # Step 3: Restore nesting [] => {}
    nested_decoded = self._nest(flatten_decoded)
    return nested_decoded


class FeaturesDictManager(_FeaturesManager, feature_lib.FeaturesDict):
  pass


class SequenceManager(_FeaturesManager, sequence_lib.Sequence):
  pass


def build_feature_manager(feature):
  """Instantiate a FeaturesManager object.

  This wrap the top level feature inside the _FeaturesManager. Because the
  top-level feature type isn't known in advance and can be FeaturesDict or
  Sequence, we dynamically construct the FeaturesManager here.

  Args:
    feature: The top-level feature, either `FeaturesDict` or `Sequence`

  Returns:
    feature_manager: The new `_FeatureManager` object with custom decoding.
  """

  if isinstance(feature, feature_lib.FeaturesDict):
    return FeaturesDictManager(feature)
  elif isinstance(feature, sequence_lib.Sequence):
    return SequenceManager(feature.feature, length=feature._length)  # pylint: disable=protected-access
  else:
    raise ValueError(
        'FeaturesManager only support FeaturesDict or Sequence at top-level. '
        'Got {}'.format(feature))


def _decode_feature(feature, example, tensor_info, serialized_info):
  """Decode a single feature."""
  sequence_rank = _get_sequence_rank(serialized_info)
  if sequence_rank == 0:
    return feature.decode_example(example)
  elif sequence_rank == 1:
    # Note: This all works fine in Eager mode (without tf.function) because
    # tf.data pipelines are always executed in Graph mode.

    # Apply the decoding to each of the individual distributed features.
    return tf.map_fn(
        feature.decode_example,
        example,
        dtype=tensor_info.dtype,
        parallel_iterations=10,
        back_prop=False,
        name='sequence_decode',
    )
  else:
    raise NotImplementedError(
        'Nested sequences not supported yet. Got: {}'.format(serialized_info)
    )


def _get_sequence_rank(serialized_info):
  """Return the number of sequence dimensions of the feature."""
  if isinstance(serialized_info, dict):
    all_sequence_rank = [s.sequence_rank for s in serialized_info.values()]
  else:
    all_sequence_rank = [serialized_info.sequence_rank]

  sequence_ranks = set(all_sequence_rank)
  if len(sequence_ranks) != 1:
    raise NotImplementedError(
        'Decoding do not support mixing sequence and context features within a '
        'single FeatureConnector. Received inputs of different sequence_rank: '
        '{}'.format(sequence_ranks)
    )
  return next(iter(sequence_ranks))
