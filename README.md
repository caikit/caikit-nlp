# Caikit NLP

Caikit-NLP is a python library providing various Natural Language Processing (NLP) capabilities built on top of [caikit](https://github.com/caikit/caikit) framework. 

## Introduction

Caikit-NLP implements concept of "task" from `caikit` framework to define (and consume) interfaces for various NLP problems and implements various "modules" to provide functionalities for these "modules". 

Capabilities provided by `caikit-nlp`:

| Task                                                | Module(s)                                      | Salient Feature(s)                                                                                                                                                                                                                                                                                                                     |
|-----------------------------------------------------|------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| TextGenerationTask                                  | 1. `PeftPromptTuning` <br> 2. `TextGeneration` | 1. Prompt Tuning, Multi-task Prompt tuning <br> 2. Fine-tuning  Both modules above provide optimized inference capability using Text Generation Inference Server                                                                                                                                                                       |
| TextClassificationTask                              | 1. `SequenceClassification`                    | 1. (Work in progress..)                                                                                                                                                                                                                                                                                                                |
| TokenClassificationTask                             | 1. `FilteredSpanClassification`                | 1. (Work in progress..)                                                                                                                                                                                                                                                                                                                |
| TokenizationTask                                    | 1. `RegexSentenceSplitter`                     | 1. Demo purposes only                                                                                                                                                                                                                                                                                                                  |
| EmbeddingTask <br> EmbeddingTasks                   | 1. `TextEmbedding`                             | 1. TextEmbedding returns a text embedding vector from a local sentence-transformers model <br> 2. EmbeddingTasks takes multiple input texts and returns a corresponding list of vectors.                                                                                                                                               
| SentenceSimilarityTask <br> SentenceSimilarityTasks | 1. `TextEmbedding`                             | 1. SentenceSimilarityTask compares one source_sentence to a list of sentences and returns similarity scores in order of the sentences. <br> 2. SentenceSimilarityTasks uses a list of source_sentences (each to be compared to same list of sentences) and returns corresponding lists of outputs.                                     |
| RerankTask <br> RerankTasks                         | 1. `TextEmbedding`                             | 1. RerankTask compares a query to a list of documents and returns top_n scores in order of relevance with indexes to the source documents and optionally returning the documents. <br> 2. RerankTasks takes multiple queries as input and returns a corresponding list of outputs. The same list of documents is used for all queries. |

## Getting Started

### Notebooks

To help you quickly get started with using Caikit, we have prepared a [Jupyter notebook](examples/Caikit_Getting_Started.ipynb) that can be run in Google Colab. Caikit-nlp is a powerful library that leverages prompt tuning and fine-tuning to add NLP domain capabilities to caikit.

### Installation

To install from git repo:

```bash
python -m venv .venv
source .venv/bin/activate
pip install git+https://github.com/caikit/caikit-nlp
```

### Bootstrapping models

`caikit_nlp` can use  Hugging Face models, allowing for direct download and bootstrapping.

For example, to use [google/flan-t5-small](https://huggingface.co/google/flan-t5-small):

```python
import os
# The env var ALLOW_DOWNLOADS has to be set to allow model downloads before importing caikit_nlp
os.environ['ALLOW_DOWNLOADS'] = "1"

import caikit_nlp

model_name = "google/flan-t5-small"
model = caikit_nlp.text_generation.TextGeneration.bootstrap(model_name)
model.save(f"{model_name}-caikit") # optionally save the model
```

### Serving models

To serve models, the following basic configuration can be used:

```yaml
# config.yml
runtime:
  library: caikit_nlp
  local_models_dir: ./models

log:
  formatter: pretty # optional: log formatter is set to json by default
```

Start the server:

```bash
env CONFIG_FILES=./config.yml python -m caikit.runtime
```

The model can now be queried at `localhost:8080` via http or at `localhost:8085` via grpc.

For example, using the http server and using curl to send a POST request:

```bash
curl --json '{
    "model_id": "flan-t5-small-caikit",
    "inputs": "At what temperature does liquid Nitrogen boil?"
}' localhost:8080/api/v1/task/text-generation
```

We get the following response:

```json
{
  "generated_text": "74 degrees F",
  "generated_tokens": 5,
  "finish_reason": "MAX_TOKENS",
  "producer_id": {
    "name": "Text Generation",
    "version": "0.1.0"
  },
  "input_token_count": 10,
  "seed": null
}
```

All the available API endpoints and protos can be dumped using [`scripts/dump_apis.sh`](/scripts/dump_apis.sh).

### Docker

To build the docker image:

```bash
python -m build --wheel
docker build -t caikit-nlp:latest .
```

A volume can be mounted at `/caikit` providing configuration and (optionally) models:

```bash
mkdir -p caikit
$EDITOR caikit/config.yml # edit as required
cp -r <path/to/models> ./caikit/models
docker run -e CONFIG_FILES=/caikit/config.yml -v $PWD/caikit/:/caikit -p 8080:8080 -p 8085:8085 python -m caikit.runtime
```

#### Serving with containers

In order to start the serving runtime:

```bash
docker run -e CONFIG_FILES=/caikit/config.yml \
    -v $PWD/caikit/:/caikit -p 8080:8080 -p 8085 \
    python -m caikit.runtime
```

Assuming the standard configuration with port `8080` for the http server and `8085` for the grpc server.

### Configuration

Configuration can be provided via environment variables or by providing a yaml configuration file thanks to [`alchemy-config`](https://github.com/IBM/alchemy-config).

For example, to set the caikit runtime, setting `RUNTIME_LIBRARY=caikit_nlp` via environment variables or providing the following yaml configuration is equivalent.

```yaml
# config.yml
runtime:
  library: caikit_nlp
```

For configuration options see `caikit_nlp`'s example config: [`config.yml`](/caikit_nlp/config/config.yml) or `caikit`'s example [`caikit.yml`](https://github.com/caikit/caikit/blob/main/caikit/config/config.yml).

## Contributing

We welcome contributions from the community! If you would like to contribute to `caikit-nlp`, please read the guidelines in the main project's [CONTRIBUTING.md](CONTRIBUTING.md) file. It includes information on submitting bug reports, feature requests, and pull requests. Make sure to follow our coding standards, [code of conduct](code-of-conduct.md), [security standards](https://github.com/caikit/community/blob/main/SECURITY.md), and documentation guidelines to streamline the contribution process.

## License

This project is licensed under the [ASFv2 License](LICENSE).

## Glossary

A list of terms that either may be unfamiliar or that have nebulous definitions based on who and where you hear them, defined for how they are used/thought of in the `caikit`/`caikit-nlp` project:

* Fine tuning - trains the base model onto new data etc; this changes the base model.
* Prompt engineering - (usually) manually crafting texts that make models do a better job that's left appended to the input text. E.g., if you wanted to do something like sentiment on movie reviews, you might come up with a prompt like The movie was: _____ and replace the _____  with the movie review you're consider to try to get something like happy/sad out of it.
* PEFT - library by Huggingface containing implementations of different tuning methods that scale well - things like prompt tuning, and MPT live there. So PEFT itself isn't an approach even though parameter efficient fine-tuning sounds like one.
Prompt tuning - learning soft prompts. This is different from prompt engineering in that you're not trying to learn tokens. Instead, you're basically trying to learn new embedded representations (sometimes called virtual tokens) that can be concatenated onto your embedded input text to improve the performance. This can work well, but also can be sensitive to initialization.
* Multitask prompt tuning (MPT) - Tries to fix some of the issues with prompt tuning by allowing you to effectively learn 'source prompts' across different tasks & leverage them to initialize your prompt tuning etc. More information on MPT can be found at: https://arxiv.org/abs/2303.02861

The important difference between fine tuning and capabilities like prompt tuning/multi-taskprompt tuning is that the latter doesn't change the base model's weights at all. So when you run inference for prompt tuned models, you can have n prompts to 1 base model, and just inject the prompt tensors you need when they're requested instead of having _n_ separate fine-tuned models.

## Runtime Performance Benchmarking

[Runtime Performance Benchmarking](./benchmarks/README.md) for tuning various models.

#### Notes

- Currently causal language models and sequence-to-sequence models are supported.
