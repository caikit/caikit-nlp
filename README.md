# Caikit NLP

Welcome to the README.md page of `caikit-nlp`! This repository contains the necessary code and resources to enable prompt tuning and fine-tuning capabilities for [caikit](https://github.com/caikit/caikit).

## Introduction

`caikit_nlp` is a [Caikit](https://github.com/caikit/caikit) library that currently provides [PEFT prompt tuning](https://github.com/huggingface/peft) and MPT (multi-task prompt tuning) functionalities.

### Getting Started

To help you quickly get started with using Caikit, we have prepared a [Jupyter notebook](examples/Caikit_Getting_Started.ipynb) that can be run in Google Colab. Caikit is a powerful library that leverages prompt tuning and fine-tuning to enhance the capabilities of our main open-source project.


### Contributing

We welcome contributions from the community! If you would like to contribute to `caikit-nlp`, please read the guidelines in the main project's [CONTRIBUTING.md](main/CONTRIBUTING.md) file. It includes information on submitting bug reports, feature requests, and pull requests. Make sure to follow our coding standards, [code of conduct](code-of-conduct.md), [security standards](https://github.com/caikit/community/blob/main/SECURITY.md), and documentation guidelines to streamline the contribution process.

### License

This project is licensed under the [ASFv2 License](LICENSE).

### Glossary

A list of terms that either may be unfamiliar or that have nebulous definitions based on who and where you hear them, defined for how they are used/thought of in the `caikit`/`caikit-nlp` project:

* Fine tuning - trains the base model onto new data etc; this changes the base model.
* Prompt engineering - (usually) manually crafting texts that make models do a better job that's left appended to the input text. E.g., if you wanted to do something like sentiment on movie reviews, you might come up with a prompt like The movie was: _____ and replace the _____  with the movie review you're consider to try to get something like happy/sad out of it.
* PEFT - library by Huggingface containing implementations of different tuning methods that scale well - things like prompt tuning, and MPT live there. So PEFT itself isn't an approach even though parameter efficient fine-tuning sounds like one.
Prompt tuning - learning soft prompts. This is different from prompt engineering in that you're not trying to learn tokens. Instead, you're basically trying to learn new embedded representations (sometimes called virtual tokens) that can be concatenated onto your embedded input text to improve the performance. This can work well, but also can be sensitive to initialization.
* Multitask prompt tuning (MPT) - Tries to fix some of the issues with prompt tuning by allowing you to effectively learn 'source prompts' across different tasks & leverage them to initialize your prompt tuning etc. More information on MPT can be found at: https://arxiv.org/abs/2303.02861

The important difference between fine tuning and stuff like prompt tuning/multiprompt tuning is that the latter doesn't change the base model's weights at all. So when you run inference for prompt tuned models, you can have n prompts to 1 base model, and just inject the prompt tensors you need when they're requested instead of having _n_ separate fine-tuned models.

#### Notes

- The data model for text generative capabilities is baked into this repository itself at `caikit_nlp/data_model/generation.py`.
- Currently causal language models and sequence-to-sequence models are supported.
