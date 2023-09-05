# Caikit NLP Tuning Benchmarks

Benchmark results for various model on various hardware configurations.

## Llama2-7b

| Date Executed |   Hardware   | Training Set | Epoch | Precision | Batch Size | Max Source Length | Training Runtime (s) | Samples Per Second | Train Steps Per Second | Loss | Notes |
|---------------|----------------------|-------|------------|-------------------|-----------|----------|--------------|--------------|--------------|------|----------------------------|
| [2023-09-06](./logs/llama2-7b/20230905_183655.output) | 1 x A100 80GB | [Glue / RTE](https://huggingface.co/datasets/glue) | 1 | bfloat16 | 6 | 4096 | 350 | 21.325 | 0.22 | 1.65 | |
| [2023-09-06](./logs/llama2-7b/20230905_184809.output) | 1 x A100 80GB | [Glue / RTE](https://huggingface.co/datasets/glue) | 1 | bfloat16 | 6 | 1096 | 350 | 21.333 | 0.22 | 1.65 | |
| [2023-09-06](./logs/llama2-7b/20230905_191650.output) | 1 x A100 80GB | [Glue / RTE](https://huggingface.co/datasets/glue) | 1 | bfloat16 | 19 | 128 | 254 | 29.332 | 0.094 | 1.938 | batch size of 20 fails CUDA OOM |
