### Tiny Models for Testing

The models in this directory were created using the [Transformers utils for creating tiny models](https://github.com/huggingface/transformers/blob/main/utils/create_dummy_models.py).

To create a new dummy model, all you need to do it clone the transformers repo and run a command like those shown below, which were used to create the artifacts checked in here.

- Bloom: `python3 utils/create_dummy_models.py --model_types bloom $OUTPUT_DIR`
- T5: `python3 utils/create_dummy_models.py --model_types t5 $OUTPUT_DIR`
- BERT: `python3 utils/create_dummy_models.py --model_types bert $OUTPUT_DIR`

This will create several dummy models; you most likely want to place the ones you need in this directory and leverage it in `__init__.py` for the fixtures.

Note: If you encounter any strangeness when running the script above, be sure to check your version of transformers in your site packages; there seem to be some dynamic imports leveraged underneath which can try to grab things from your site packages instead of the direct source, which can cause some problems if the cloned version of the code is very different.
