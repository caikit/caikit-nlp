# Caikit Prompt Tuning (PT)

`caikit_pt` is a [Caikit](https://github.com/caikit/caikit) library that currently provides [PEFT prompt tuning](https://github.com/huggingface/peft) and MPT (multi-task prompt tuning) functionalities.

More information on MPT can be found at: https://arxiv.org/abs/2303.02861

Currently causal language models and sequence-to-sequence models are supported.

#### Notes

- The data model for text generative capabilities is baked into this repository itself at `caikit_pt/data_model/generation.py`.
