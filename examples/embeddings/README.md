# Set up and run locally caikit embeddings server

#### Setting Up Virtual Environment using Python venv

For [(venv)](https://docs.python.org/3/library/venv.html), make sure you are in an activated `venv` when running `python` in the example commands that follow. Use `deactivate` if you want to exit the `venv`.

```shell
python3 -m venv venv
source venv/bin/activate
```

### Models

For this tutorial, you can download [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), to do that you need to follow the steps to clone and use `git lfs` to get all the models files:

```shell
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install

git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
```

To create a model configuration and artifacts, the best practice is to run the module's bootstrap() and save() methods.  This will:

* Load the model by name (from Hugging Face hub or repository) or from a local directory. The model is loaded using the sentence-transformers library.
* Save a config.yml which:
  * Ties the model to the module (with a module_id GUID)
  * Sets the artifacts_path to the default "artifacts" subdirectory
  * Saves the model in the artifacts subdirectory
* Check an example of the folder structure at [models_](./models_/)

> For the reranker service, models supported are bi-encoder and are the same used by the other embeddings tasks.

This can be achieved by the following lines of code, using BGE as example model:

```python
import os
os.environ['ALLOW_DOWNLOADS'] = "1"

import caikit_nlp
model_name = "BAAI/bge-large-en-v1.5"
model = caikit_nlp.text_embedding.EmbeddingModule.bootstrap(model_name)
model.save(f"{model_name}-caikit") 
```

To avoid overwriting your files, the save() will return an error if the output directory already exists. You may want to use a temporary name. After success, move the output directory to a `<model-id>` directory under your local models dir.

### Environment variables

These are the set of variables/params related to the environment which embeddings will be run:

```bash
# use IPEX optimization
IPEX_OPTIMIZE: 'true'

# use "xpu" for IPEX on GPU instead of IPEX on CPU
USE_XPU: 'false'

# IPEX performs best with autocast using bfloat16
BFLOAT16: '1'

# use Mac chip
USE_MPS: 'false'

# use Pytorch compile
PT2_COMPILE: 'false'
```

### Starting the Caikit Runtime

Run caikit-runtime configured to use the caikit-nlp library. Set up the following environment variables:

```bash
# set where the runtime should look for the models
export RUNTIME_LOCAL_MODELS_DIR=/models_

# load the models from the path set up at previous var
export RUNTIME_LAZY_LOAD_LOCAL_MODELS=true

# set the runtime
export RUNTIME_LIBRARY='caikit_nlp'
```

In one terminal, start the runtime server:

```bash
source venv/bin/activate
pip install -r requirements.txt
caikit-runtime
```

To run the library locally:

```bash
pip install caikit-nlp@file:///<path-to-your-local-caikit_nlp-clone-repo>/caikit-nlp
python -m caikit.runtime
```

### Embedding retrieval example Python client

In another terminal, run the example client code to retrieve embeddings.

```shell
source venv/bin/activate
MODEL=<model-id> python embeddings.py
```

The client code calls the model and queries for embeddings using 2 example sentences.

You should see output similar to the following:

```ShellSession
$ python embeddings.py
INPUT TEXTS:  ['test first sentence', 'another test sentence']
OUTPUT: {
  {
  "results": [
    [
      -0.17895537614822388,
      0.03200146183371544,
      -0.030327674001455307,
      ...
    ],
    [
      -0.17895537614822388,
      0.03200146183371544,
      -0.030327674001455307,
      ...
    ]
  ],
  "producerId": {
    "name": "EmbeddingModule",
    "version": "0.0.1"
  },
  "inputTokenCount": "9"
  }
}
LENGTH:  2  x  384
```

### Sentence similarity example Python client

In another terminal, run the client code to infer sentence similarity.

```shell
source venv/bin/activate
MODEL=<model-id> python sentence_similarity.py
```

The client code calls the model and queries sentence similarity using 1 source sentence and 2 other sentences (hardcoded in sentence_similarity.py). The result produces the cosine similarity score by comparing the source sentence with each of the other sentences.

You should see output similar to the following:

```ShellSession
$ python sentence_similarity.py   
SOURCE SENTENCE:  first sentence
SENTENCES:  ['test first sentence', 'another test sentence']
OUTPUT: {
  "result": {
    "scores": [
      1.0000001192092896
    ]
  },
  "producerId": {
    "name": "EmbeddingModule",
    "version": "0.0.1"
  },
  "inputTokenCount": "9"
}
```

### Reranker example Python client

In another terminal, run the client code to execute the reranker task using both gRPC and REST.

```shell
source venv/bin/activate
MODEL=<model-id> python reranker.py
```

You should see output similar to the following:

```ShellSession
$ python reranker.py
======================
TOP N:  3
QUERIES:  ['first sentence', 'any sentence']
DOCUMENTS:  [{'text': 'first sentence', 'title': 'first title'}, {'_text': 'another sentence', 'more': 'more attributes here'}, {'text': 'a doc with a nested metadata', 'meta': {'foo': 'bar', 'i': 999, 'f': 12.34}}]
======================
RESPONSE from gRPC:
===
QUERY:  first sentence
  score: 0.9999997019767761  index: 0  text: first sentence
  score: 0.7350112199783325  index: 1  text: another sentence
  score: 0.10398174077272415  index: 2  text: a doc with a nested metadata
===
QUERY:  any sentence
  score: 0.6631797552108765  index: 0  text: first sentence
  score: 0.6505964398384094  index: 1  text: another sentence
  score: 0.11903437972068787  index: 2  text: a doc with a nested metadata
===================
RESPONSE from HTTP:
{
    "results": [
        {
            "query": "first sentence",
            "scores": [
                {
                    "document": {
                        "text": "first sentence",
                        "title": "first title"
                    },
                    "index": 0,
                    "score": 0.9999997019767761,
                    "text": "first sentence"
                },
                {
                    "document": {
                        "_text": "another sentence",
                        "more": "more attributes here"
                    },
                    "index": 1,
                    "score": 0.7350112199783325,
                    "text": "another sentence"
                },
                {
                    "document": {
                        "text": "a doc with a nested metadata",
                        "meta": {
                            "foo": "bar",
                            "i": 999,
                            "f": 12.34
                        }
                    },
                    "index": 2,
                    "score": 0.10398174077272415,
                    "text": "a doc with a nested metadata"
                }
            ]
        },
        {
            "query": "any sentence",
            "scores": [
                {
                    "document": {
                        "text": "first sentence",
                        "title": "first title"
                    },
                    "index": 0,
                    "score": 0.6631797552108765,
                    "text": "first sentence"
                },
                {
                    "document": {
                        "_text": "another sentence",
                        "more": "more attributes here"
                    },
                    "index": 1,
                    "score": 0.6505964398384094,
                    "text": "another sentence"
                },
                {
                    "document": {
                        "text": "a doc with a nested metadata",
                        "meta": {
                            "foo": "bar",
                            "i": 999,
                            "f": 12.34
                        }
                    },
                    "index": 2,
                    "score": 0.11903437972068787,
                    "text": "a doc with a nested metadata"
                }
            ]
        }
    ],
     "producerId": {
    "name": "EmbeddingModule",
    "version": "0.0.1"
    },
    "inputTokenCount": "9"
}
```