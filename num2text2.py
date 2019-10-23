# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
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

"""Data generators for translation data-sets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import translate
from tensor2tensor.data_generators import wiki_lm
from tensor2tensor.utils import registry
from tensor2tensor.models.transformer import transformer_base, transformer_base_multistep8

from .text_gen import next_pair
from re_map import Processor
import re
import io
import tensorflow as tf


_NUM2TEXT_DATASETS = [
    [
        "https://raw.githubusercontent.com/aleksas/liepa_dataset/master/other/stressed/__final_1.txt",  # pylint: disable=line-too-long
        ("__final_1.txt")
    ]
]



def _get_num2text_dataset(directory, filename):
  """Extract the WMT en-de corpus `filename` to directory unless it's there."""
  train_path = os.path.join(directory, filename)
  if not (tf.gfile.Exists(train_path + "__final_1.txt")):
    url = _NUM2TEXT_DATASETS[0][0]
    generator_utils.maybe_download(directory, "__final_1.txt", url)
  return train_path


@registry.register_problem
class NumToText(translate.TranslateProblem):
  @property
  def var1(self):
    return 5

  @property
  def vocab_type(self):
    return text_problems.VocabType.SUBWORD

  @property
  def is_generate_per_split(self):
    return False

  @property
  def dataset_splits(self):
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 90,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 5,
    }, {
        "split": problem.DatasetSplit.TEST,
        "shards": 5,
    }]

  @property
  def approx_vocab_size(self):
    return 2**13  # 256

  def source_data_files(self, dataset_split):
      return _NUM2TEXT_DATASETS

  def get_pairs(self, data_dir, tmp_dir):
    train_path = _get_num2text_dataset(tmp_dir, '')

    with io.open(train_path + "__final_1.txt", mode="r", encoding="utf-8") as fp:
      with Processor(fp.read()) as processor:
        re_clean = re.compile(r'[~`\^]')
        processor.process(
            pattern=r'([A-ZĄ-ža-zą-ž`~^]+)',
            replacement_map={ 1: lambda x : re_clean.sub('', x) }
        )

        processor.swap()
      
      
      for source_span, target_span in next_pair(processor.span_map, 250, 250):
        source_text = processor.text[source_span[0]:source_span[1]]
        target_text = processor.processed_text[target_span[0]:target_span[1]]

        yield source_text, target_text


  def generate_text_for_vocab(self, data_dir, tmp_dir):      
    for inputs, targets in self.get_pairs(data_dir, tmp_dir):
      yield inputs
      yield targets

  def generate_samples(self, data_dir, tmp_dir, dataset_split):      
      for inputs, targets in self.get_pairs(data_dir, tmp_dir):
        yield {"inputs": inputs, "targets": targets}

@registry.register_problem
class NumToText1K(NumToText):
  @property
  def approx_vocab_size(self):
    return 2**10  # 1024

@registry.register_hparams
def transformer_base_bs1k_lrc1():
  """HParams for simulating 8 GPUs with MultistepAdam optimizer."""
  hparams = transformer_base()
  hparams.batch_size = 1024
  hparams.eval_drop_long_sequences=True
  hparams.learning_rate_constant = 1.0
  return hparams

@registry.register_hparams
def transformer_base_bs94_lrc1():
  """HParams for simulating 8 GPUs with MultistepAdam optimizer."""
  hparams = transformer_base_bs1k_lrc1()
  hparams.batch_size = 9400
  return hparams

@registry.register_hparams
def transformer_base_multistep12_bs94_lrws10():
  """HParams for simulating 8 GPUs with MultistepAdam optimizer."""
  """Blueu<0.97 rouge_2<0.96 rouge_L~0.97 sequence_acc<0.995 acc=1"""
  hparams = transformer_base_multistep8()
  hparams.batch_size = 9400
  hparams.eval_drop_long_sequences=True
  hparams.learning_rate_warmup_steps=10000
  hparams.optimizer_multistep_accumulate_steps=12
  return hparams

@registry.register_hparams
def transformer_base_bs94_lrc1_do2():
  """HParams for simulating 8 GPUs with MultistepAdam optimizer."""
  hparams = transformer_base_bs94_lrc1()
  
  hparams.attention_dropout = 0.2
  hparams.relu_dropout = 0.2
  hparams.layer_prepostprocess_dropout = 0.2

  hparams.attention_dropout_broadcast_dims = "0,1"  # batch, heads
  hparams.relu_dropout_broadcast_dims = "1"  # length
  hparams.layer_prepostprocess_dropout_broadcast_dims = "1"

  return hparams

@registry.register_hparams
def transformer_base_bs94_lrc1_do3():
  """HParams for simulating 8 GPUs with MultistepAdam optimizer."""
  hparams = transformer_base_bs94_lrc1_do2()
  
  hparams.attention_dropout = 0.3
  hparams.relu_dropout = 0.3
  hparams.layer_prepostprocess_dropout = 0.3
  return hparams


@registry.register_hparams
def transformer_base_bs94_lrc1_do4():
  """HParams for simulating 8 GPUs with MultistepAdam optimizer."""
  hparams = transformer_base_bs94_lrc1_do2()
  
  hparams.attention_dropout = 0.4
  hparams.relu_dropout = 0.4
  hparams.layer_prepostprocess_dropout = 0.4
  return hparams

@registry.register_hparams
def transformer_base_bs94_lrc1_do5():
  """HParams for simulating 8 GPUs with MultistepAdam optimizer."""
  hparams = transformer_base_bs94_lrc1_do2()
  
  hparams.attention_dropout = 0.5
  hparams.relu_dropout = 0.5
  hparams.layer_prepostprocess_dropout = 0.5
  return hparams


@registry.register_hparams
def transformer_base_bs94_lrc1_do3_b():
  """HParams for simulating 8 GPUs with MultistepAdam optimizer."""
  hparams = transformer_base_bs94_lrc1_do3()
  return hparams


@registry.register_hparams
def transformer_base_bs94_lrc1_do4_b():
  hparams = transformer_base_bs94_lrc1_do4()
  return hparams


@registry.register_hparams
def transformer_base_bs94_lrc1_do4_c():
  hparams = transformer_base_bs94_lrc1_do4()
  return hparams

@registry.register_hparams
def transformer_base_bs94_1k_lrc1_do4_d():
  hparams = transformer_base_bs94_lrc1_do4()
  return hparams

@registry.register_hparams
def transformer_base_bs94_lrc1_do4_e():
  hparams = transformer_base_bs94_lrc1_do4()
  return hparams

@registry.register_hparams
def transformer_base_bs94_lrc1_do4_f():
  hparams = transformer_base_bs94_lrc1_do4()
  hparams.batch_size = 8800
  return hparams

