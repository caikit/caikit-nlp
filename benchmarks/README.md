# Caikit NLP Tuning Benchmarks

Benchmark results for various model on various hardware configurations.

## Llama2-7b

| Date Executed | Hardware | Training Set | Precision | Epoch | Batch Size | Max Source Length | Training Runtime (s) | Samples Per Second | Train Steps Per Second | Loss |
|---------------|--------------|-------|------------|-------------------|-----------|----------|--------------|--------------|--------------|------|
| [2023-09-06](./logs/llama2-7b/20230905_183655.output) | A100 80GB | [Glue / RTE](https://huggingface.co/datasets/glue) | 1 | bfloat16 | 1 | 6 | 4096 | 350 | 21.325 | 0.22 | 1.65 |
