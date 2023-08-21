# Caikit NLP

Caikit-NLP is a python library providing various Natural Language Processing (NLP) capabilities built on top of [caikit](https://github.com/caikit/caikit) framework. 

## Introduction

Caikit-NLP implements concept of "task" from `caikit` framework to define (and consume) interfaces for various NLP problems and implements various "modules" to provide functionalities for these "modules". 

Capabilities provided by `caikit-nlp`:

| Task                 | Module(s)                                 | Salient Feature(s)                                                                                                                                          |
|----------------------|-------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Text Generation      | 1. `PeftPromptTuning` <br> 2. `TextGeneration` | 1. Prompt Tuning, Multi-task Prompt tuning <br> 2. Fine-tuning  Both modules above provide optimized inference capability using Text Generation Inference Server |
| Text Classification  | 1. `SequenceClassification`               | 1. (Work in progress..)                                                                                                                                     |
| Token Classification | 1. `FilteredSpanClassification`           | 1. (Work in progress..)                                                                                                                                     |
| Tokenization         | 1. `RegexSentenceSplitter`                | 1. Demo purposes only                                                                                                                                       |
| Embedding         | [COMING SOON]                | [COMING SOON]                                                                                                                                       |

### Getting Started

To help you quickly get started with using Caikit, we have prepared a [Jupyter notebook](examples/Caikit_Getting_Started.ipynb) that can be run in Google Colab. Caikit-nlp is a powerful library that leverages prompt tuning and fine-tuning to add NLP domain capabilities to caikit.


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

The important difference between fine tuning and capabilities like prompt tuning/multi-taskprompt tuning is that the latter doesn't change the base model's weights at all. So when you run inference for prompt tuned models, you can have n prompts to 1 base model, and just inject the prompt tensors you need when they're requested instead of having _n_ separate fine-tuned models.

#### Notes

- Currently causal language models and sequence-to-sequence models are supported.
