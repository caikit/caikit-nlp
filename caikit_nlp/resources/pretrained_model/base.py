# Copyright The Caikit Authors
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

# Standard
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple, Type, Union
import json
import os

# Third Party
from torch.utils.data import IterableDataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from transformers.models.auto.auto_factory import _BaseAutoModelClass
import torch

# First Party
from caikit import get_config
from caikit.core.modules import ModuleBase, ModuleConfig, ModuleSaver
from caikit.core.toolkit import error_handler
import alog

# Local
from ...data_model import PromptOutputModelType
from ...toolkit.data_type_utils import get_torch_dtype, str_to_torch_dtype

log = alog.use_channel("HFRBAS")
error = error_handler.get(log)


class PretrainedModelBase(ABC, ModuleBase):
    """Common abstractions and requirements for pretrained model resources"""

    _TOK_ARTIFACTS_CONFIG_KEY = "tokenizer_artifacts"
    _MODEL_ARTIFACTS_CONFIG_KEY = "model_artifacts"
    _LEFT_PAD_MODEL_TYPES = ("gpt", "opt", "bloom")

    ## Abstract Interface ######################################################

    @classmethod
    @property
    @abstractmethod
    def MODEL_TYPE(cls) -> Type[_BaseAutoModelClass]:
        """All classes must have a class property declaring the type of HF model
        they support
        """

    @classmethod
    @property
    @abstractmethod
    def TASK_TYPE(cls) -> str:
        """All classes must have indicate the PEFT task type that they use"""

    @classmethod
    @property
    @abstractmethod
    def SUPPORTED_MODEL_TYPES(cls) -> str:
        """All classes must indicate the model types supported by the resource"""

    ## Shared Implementation ###################################################

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        model: _BaseAutoModelClass,
        model_name: str,
        torch_dtype: torch.dtype,
    ):
        """Initialize with an in-memory handle to a model"""
        super().__init__()
        self._tokenizer = tokenizer
        self._model = model
        self._model_name = model_name
        self._torch_dtype = torch_dtype

    @property
    def model(self) -> _BaseAutoModelClass:
        """Get access to the underlying causal LM"""
        return self._model

    @property
    def tokenizer(self) -> AutoTokenizer:
        """Get access to the underlying tokenizer"""
        return self._tokenizer

    def get_config(self):
        """Function to return model config from transformer model"""
        # This funky json.load step is just to make sure any non-string
        # keys gets converted to string automatically otherwise
        # it will throw error when we try to save them using ModuleSaver.
        # if it was not for this, we could have just used self._module.config.to_dict()
        return json.loads(self.model.config.to_json_string())

    @classmethod
    def bootstrap(
        cls,
        model_name: str,
        tokenizer_name: Optional[str] = None,
        padding_side: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        **kwargs,
    ) -> "HFAutoSequenceClassifier":
        """Bootstrap from a huggingface model

        This method takes the raw HF model name / path and bootstraps the
        tokenizer in memory. If downloads are enabled, the model name can be any
        model name from HF, otherwise it must be a path to an on-disk model or
        the name of a pre-cached model.

        Args:
            model_name (str)
                The name/path of the HF sequence classifier model
            tokenizer_name (Optional[str])
                The name/path of the HF tokenizer model (matches model_name if
                not given)
            padding_side (Optional[str])
                The padding side for the tokenizer. Found by convention if not
                given.
            torch_dtype: (Optional[Union[torch.dtype, str]])
                Data type to load the model as; if no value is provided, we pull
                torch_dtype from config.
            **kwargs
                Additional keyword args to pass to from_pretrained
                (e.g. return_dict=True)

        Returns:
            model HFAutoSequenceClassifier
                The loaded resource model
        """
        # Default tokenizer to model if downloading with a model name rather
        # than a path
        error.value_check(
            "<NLP12813423E>",
            not os.path.isdir(model_name) or tokenizer_name is not None,
            "Must provide path to tokenizer if model_name is a path",
        )
        torch_dtype = get_torch_dtype(torch_dtype)
        if tokenizer_name is None:
            tokenizer_name = model_name
        if not os.path.isdir(tokenizer_name) and tokenizer_name != model_name:
            log.warning(
                "Bootstrapping with mismatched tokenizer (%s) / model (%s)",
                tokenizer_name,
                model_name,
            )

        # Figure out the right padding side based on the name of the HF model
        # NOTE: This matches models whose name includes the left-pad types as a
        #   substring and not just as an exact match.
        if padding_side is None:
            padding_side = (
                "left"
                if any(k in model_name for k in cls._LEFT_PAD_MODEL_TYPES)
                else "right"
            )

        # Load the tokenizer and set up the pad token if needed
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            local_files_only=not get_config().allow_downloads,
            padding_side=padding_side,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Load the model
        model = cls.MODEL_TYPE.from_pretrained(
            model_name,
            local_files_only=not get_config().allow_downloads,
            torch_dtype=torch_dtype,
            **kwargs,
        )
        log.debug4("Model Details: %s", model)

        # Create the class instance
        inst_model_name = os.path.split(model_name)[-1]
        return cls(
            tokenizer=tokenizer,
            model=model,
            model_name=inst_model_name,
            torch_dtype=torch_dtype,
        )

    @classmethod
    def load(cls, model_path: str) -> Type["PretrainedModelBase"]:
        """Load from a saved resource model"""
        model_path = os.path.abspath(model_path)
        config = ModuleConfig.load(model_path)
        tok_abs_path = os.path.join(model_path, config[cls._TOK_ARTIFACTS_CONFIG_KEY])
        model_abs_path = os.path.join(
            model_path, config[cls._MODEL_ARTIFACTS_CONFIG_KEY]
        )
        error.dir_check("<NLP12813455E>", tok_abs_path)
        error.dir_check("<NLP12813443E>", model_abs_path)
        res = cls.bootstrap(
            tokenizer_name=tok_abs_path,
            model_name=model_abs_path,
            padding_side=config.padding_side,
            torch_dtype=str_to_torch_dtype(config.torch_dtype),
        )
        return res

    def save(
        self, model_path: str, tokenizer_dirname: str = "", base_model_dirname: str = ""
    ):
        """Save the in-memory model to the given path"""
        saver = ModuleSaver(
            self,
            model_path=model_path,
        )
        tok_rel_path, tok_abs_path = saver.add_dir(tokenizer_dirname)
        model_rel_path, model_abs_path = saver.add_dir(base_model_dirname)
        with saver:
            saver.update_config(
                {
                    self._TOK_ARTIFACTS_CONFIG_KEY: tok_rel_path,
                    self._MODEL_ARTIFACTS_CONFIG_KEY: model_rel_path,
                    "padding_side": self.tokenizer.padding_side,
                    "model_name": self._model_name,
                    # Grab the torch property for the dtype so that we can rebuild from a str.
                    "torch_dtype": str(self._torch_dtype).rsplit(".", maxsplit=1)[-1],
                }
            )
            self.tokenizer.save_pretrained(tok_abs_path)
            self.model.save_pretrained(model_abs_path)

    def get_trainer(
        self,
        train_dataset: IterableDataset,
        eval_dataset: Union[IterableDataset, None] = None,
        optimizers=(None, None),
        **kwargs,
    ):
        """
        Args:
            **kwargs: arguments supported by HF TrainingArguments:
            https://huggingface.co/docs/transformers/v4.30.0/en/main_classes/trainer#transformers.TrainingArguments

        NOTE: following parameters are not supported currently:
            1. model_init
            2. compute_metrics
            3. callbacks
            4. preprocess_logits_for_metrics
        """

        training_args = TrainingArguments(**kwargs)

        data_collator = self._get_data_collator(**kwargs)

        trainer_arguments = {
            "train_dataset": train_dataset,
            "data_collator": data_collator,
            "tokenizer": self._tokenizer,
            "optimizers": optimizers,
            "eval_dataset": eval_dataset,
        }

        return Trainer(self._model, training_args, **trainer_arguments)

    def _get_data_collator(self, **kwargs):
        """Function to return appropriate data collator based on resource.

        The default implementation of the base resource uses
        DataCollatorWithPadding which will dynamically pad the inputs received.

        Args:
            **kwargs:
                All the keyword arguments passed to this function
                will get filtered out to appropriate ones that are
                applicable to implemented data collator.
        Returns:
            transformers.DataCollator
        """

        applicable_args = ["max_length", "pad_to_multiple_of"]
        collator_kwargs = {key: kwargs[key] for key in applicable_args if key in kwargs}

        return DataCollatorWithPadding(
            tokenizer=self._tokenizer, padding=True, **collator_kwargs
        )

    # pylint: disable=unused-argument
    @classmethod
    def get_num_transformers_submodules(
        cls, output_model_types: List[PromptOutputModelType]
    ):
        """Return number of applicable transformer submodules"""
        return 1

    @classmethod
    @abstractmethod
    def build_task_tokenize_function(
        cls,
        tokenizer: "AutoTokenizer",
        max_source_length: int,
        max_target_length: int,
        verbalizer: str,
        task_ids: Union[None, int] = None,
    ) -> Tuple[Callable, bool]:
        """Builds tokenizer functions which can be mapped over train streams to process
        data which can then be easily passed to a DataLoader for different model types.

        Args:
            tokenizer: AutoTokenizer
                Model tokenizer to be used in preprocessing, i.e., when we iterate over our data.
            max_source_length: int
                Max length of sequences being considered.
            max_target_length: int
                Max length of target sequences being predicted.
            verbalizer: str
                Verbalizer template to be used for formatting data. This template may use brackets
                to indicate where fields from the data model TrainGenerationRecord must be rendered.
            task_ids: Union[None, int]
                Task id corresponding particular task for multi-task prompt tuning.
                NOTE: Only required for MPT (Multi-task prompt tuning)
                Default: None

        Returns:
            Tuple(Callable, bool)
                Mappable tokenize function to be applied to a training stream and bool indicating
                whether or not the stream needs to be unwrapped, i.e., each sample yields a stream
                of 1+ samples.
        """
