# Caikit NLP Runtime Performance Benchmarks

Runtime performance benchmarking results for various model on various hardware configurations.

## Llama2-7b

| Date Executed |   Hardware   | Training Set | Epoch | Precision | Batch Size | Max Source Length | Training Runtime (s) | Samples Per Second | Train Steps Per Second | Loss |        Notes         |
|---|---|---------------|---|---|:---:|---|------------| --- |---|---|---|
| [2023-09-05](./logs/llama2-7b/20230905_183655.output) | 1 x A100 80GB | [Glue / RTE](https://huggingface.co/datasets/glue) | 1 | bfloat16 | 6 | 4096 | 350 | 21.325 | 0.22 | 1.65 |     4096 is the context size for Llama2     |
| [2023-09-05](./logs/llama2-7b/20230905_184809.output) | 1 x A100 80GB | [Glue / RTE](https://huggingface.co/datasets/glue) | 1 | bfloat16 | 6 | 1024 | 350 | 21.333 | 0.22 | 1.65 | batch size of 7 fails CUDA OOM |
| [2023-09-06](./logs/llama2-7b/20230906_135211.output) | 1 x A100 80GB | [Glue / RTE](https://huggingface.co/datasets/glue) | 1 | bfloat16 | 6 | 512 | 348 | 21.44 | 0.22 | 1.65 | batch size of 7 fails CUDA OOM |
| [2023-09-05](./logs/llama2-7b/20230905_194133.output) | 1 x A100 80GB | [Glue / RTE](https://huggingface.co/datasets/glue) | 1 | bfloat16 | 8 | 256 | 356 | 20.939 | 0.16 | 1.70 | batch size of 9 fails CUDA OOM |
| [2023-09-05](./logs/llama2-7b/20230905_191650.output) | 1 x A100 80GB | [Glue / RTE](https://huggingface.co/datasets/glue) | 1 | bfloat16 | 19 | 128 | 254 | 29.332 | 0.09 | 1.94 | batch size of 20 fails CUDA OOM |
