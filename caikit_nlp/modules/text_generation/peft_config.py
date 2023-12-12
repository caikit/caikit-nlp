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
from enum import Enum
import os
import re

# Third Party
from peft import LoraConfig, MultitaskPromptTuningInit, PromptTuningConfig
from transformers import AutoConfig

# First Party
from caikit import get_config
from caikit.core import error_handler
import alog

# Local
from ...data_model import PromptOutputModelType
from ...resources.pretrained_model import PretrainedModelBase
from ...toolkit.data_type_utils import get_torch_dtype
from ...toolkit.verbalizer_utils import is_valid_verbalizer

# NOTE: We do not allow all the methods exposed by MPT / PT, such as `EXACT_SOURCE_TASK`
# since those are for experimental use and would not be useful / applicable
# for end-user use-cases
allowed_tuning_init_methods = [
    "TEXT",
    "RANDOM",
    "ONLY_SOURCE_SHARED",
    "AVERAGE_SOURCE_TASKS",
]

log = alog.use_channel("PFT_CNFG_TLKT")
error = error_handler.get(log)

SOURCE_DIR_VALIDATION_REGEX = re.compile(r"^[-a-zA-Z_0-9\/\.]+")
# 🤮 FIXME: This two dot regex is added as a way to avoid expressions like ..
# giving access to un-intended directories. But this is an ugly hack
# and we need to figure out better solution or better regex
TWO_DOTS_REGEX = re.compile(r"(\.\.)+")


class TuningType(str, Enum):
    PROMPT_TUNING = "PROMPT_TUNING"
    MULTITASK_PROMPT_TUNING = "MULTITASK_PROMPT_TUNING"
    LORA = "LORA"
    # MULTITASK_PREFIX_TUNING = "MULTITASK_PREFIX_TUNING"
    # P_TUNING = "P_TUNING"
    # PREFIX_TUNING = "PREFIX_TUNING"


def resolve_base_model(base_model, cls, torch_dtype):
    if isinstance(base_model, str):

        error.value_check(
            "<NLP66932773E>",
            re.fullmatch(SOURCE_DIR_VALIDATION_REGEX, base_model)
            and not re.search(TWO_DOTS_REGEX, base_model),
            "invalid characters in base_model name",
        )
        if get_config().base_models_dir:

            base_model_full_path = os.path.join(
                get_config().base_models_dir, base_model
            )
            if os.path.exists(base_model_full_path):
                base_model = base_model_full_path

        model_config = AutoConfig.from_pretrained(
            base_model, local_files_only=not get_config().allow_downloads
        )

        resource_type = None
        for resource in cls.supported_resources:
            if model_config.model_type in resource.SUPPORTED_MODEL_TYPES:
                resource_type = resource
                break

        if not resource_type:
            error(
                "<NLP61784225E>",
                "{} model type is not supported currently!".format(
                    model_config.model_type
                ),
            )
        log.debug("Bootstrapping base resource [%s]", base_model)
        base_model = resource_type.bootstrap(base_model, torch_dtype=torch_dtype)
    return base_model


def get_peft_config(
    tuning_type,
    tuning_config,
    base_model,
    cls=None,
    torch_dtype=None,
    verbalizer="{{input}}",
):

    if isinstance(tuning_type, str):
        error.value_check(
            "<NLP65714994E>",
            tuning_type in TuningType._member_names_,
            f"Invalid tuning type [{tuning_type}]. Allowed types: "
            f"[{TuningType._member_names_}]",
        )
        tuning_type = TuningType(tuning_type)

    error.type_check("<NLP65714993E>", TuningType, tuning_type=tuning_type)

    if tuning_type not in TuningType._member_names_:
        raise NotImplementedError("{} tuning type not supported!".format(tuning_type))

    error.type_check("<NLP65714919E>", PretrainedModelBase, base_model=base_model)

    # Validate if tuned output model type is compatible with base model or not
    output_model_types = _get_output_types(tuning_config, base_model)

    # NOTE: Base model is a resource at this point
    task_type = base_model.TASK_TYPE

    error.value_check(
        "<NLP30542004E>",
        len(output_model_types) <= base_model.MAX_NUM_TRANSFORMERS,
        f"Too many output model types. Got {len(output_model_types)}, "
        f"maximum {base_model.MAX_NUM_TRANSFORMERS}",
    )

    if verbalizer:
        log.warning(
            "<NLP21323085W>",
            "verbalizer parameter is DEPRECATED for this function \
            and will be removed in future. \
            This parameter is also not getting used in creation of peft config",
        )
        # Ensure that our verbalizer is a string and
        # will not render to a hardcoded string
        # TODO: This check should happen in prompt tuning module and not here
        error.value_check(
            "<NLP83837412E>",
            is_valid_verbalizer(verbalizer),
            "Provided verbalizer is an invalid type or has no renderable placeholders",
        )

    if torch_dtype:
        log.warning(
            "<NLP16173085W>",
            "torch_dtype parameter is DEPRECATED for this function \
            and will be removed in future. \
            This parameter is also not getting used in creation of peft config",
        )
        torch_dtype = get_torch_dtype(torch_dtype)
    print("type", tuning_type)
    if tuning_type in [
        TuningType.PROMPT_TUNING,
        TuningType.MULTITASK_PROMPT_TUNING,
    ]:
        peft_config = _create_prompt_tuning_config(
            tuning_type, tuning_config, cls, base_model, task_type, output_model_types
        )
    else:
        # we only have Lora besides other two for now
        peft_config = _create_lora_config(tuning_config, task_type)

    return task_type, output_model_types, peft_config, tuning_type


def _get_output_types(tuning_config, base_model):
    "Validate and return output_model_types"
    # Validate if tuned output model type is compatible with base model or not
    if not tuning_config.output_model_types:
        output_model_types = base_model.PROMPT_OUTPUT_TYPES
    else:
        # If the first element is not PromptOutputModelType, assume the entire list
        # isn't and convert
        if not isinstance(tuning_config.output_model_types[0], PromptOutputModelType):
            output_model_types = []
            for output_type in tuning_config.output_model_types:
                output_model_types.append(PromptOutputModelType(output_type))
        else:
            output_model_types = tuning_config.output_model_types
        error.value_check(
            "<NLP36947542E>",
            all(
                output_type in base_model.PROMPT_OUTPUT_TYPES
                for output_type in output_model_types
            ),
            "{} not supported for base model type {}".format(
                output_model_types, base_model.MODEL_TYPE
            ),
        )
    return output_model_types


def _filter_params_for_prompt_config(prompt_config, params):
    """Utility function to filter out required parameters for prompt_config
    from `params`

    Args:
        prompt_config: PromptTuningConfig
            Tuning config type, eg:, PromptTuningConfig
        params: dict
            Dictionary containing all the input training params

    Returns:
        dict:
            Dictionary containing required params for prompt_config
    """
    # Inspect the underlying dataclass fileds; we do this because the common super class
    # used for multi/vanilla prompt/prefix tuning is a DataClass; we can't use __dict__
    # because the dataclass fields are omitted.
    allowed_keys = list(prompt_config.__dataclass_fields__.keys())
    allowed_params = dict(filter(lambda x: x[0] in allowed_keys, params.items()))
    log.info(
        "<NLP18184771I>",
        "[{}] config params not supported by provided tuning type!".format(
            params.keys() - allowed_params.keys()
        ),
    )
    return allowed_params


def _create_prompt_tuning_config(
    tuning_type, tuning_config, cls, base_model, task_type, output_model_types
) -> PromptTuningConfig:
    """Creates Huggingface PromptTuningConfig from Caikit tuning configuration."""

    if tuning_config.prompt_tuning_init_method:
        # NOTE: GK-APR-5-2023
        # MultitaskPromptTuningInit and MultitaskPrefixTuningInit are same at the
        # time of writing, which is a superset of PromptTuningInit
        init_method = tuning_config.prompt_tuning_init_method

        error.value_check(
            "<NLP11848053E>",
            init_method in allowed_tuning_init_methods,
            f"Init method [{init_method}] not in allowed init methods: "
            f"[{allowed_tuning_init_methods}]",
        )

        init_method = MultitaskPromptTuningInit(init_method)
        log.info("Using initialization method [%s]", init_method)

        # If init method provided relates to one that requires source model,
        # make sure the source prompt model is provided.
        if init_method in [
            MultitaskPromptTuningInit.AVERAGE_SOURCE_TASKS,
            MultitaskPromptTuningInit.ONLY_SOURCE_SHARED,
        ]:
            # NOTE: prompt_tuning_init_source_model is currently a path. In future
            # we will replace this with caikit.resources to properly cataloging these
            error.type_check(
                "<NLP89108490E>",
                str,
                prompt_tuning_init_source_model=tuning_config.prompt_tuning_init_source_model,
            )
            tuning_config.prompt_tuning_init_source_model = os.path.join(
                get_config().source_prompt_base,
                tuning_config.prompt_tuning_init_source_model,
            )

            error.file_check(
                "<NLP96030210E>", tuning_config.prompt_tuning_init_source_model
            )
            log.debug(
                "Validated tuning source prompt [%s]",
                tuning_config.prompt_tuning_init_source_model,
            )

    # Take tokenizer name/path from the model
    tokenizer_name_or_path = base_model.model.config._name_or_path

    # Build the peft config; this is how we determine that we want a sequence classifier.
    # If we want more types, we will likely need to map this to data model outputs etc.

    # NOTE: We currently only support TEXT as init type, this is to later only easily
    # switch to MPT
    prompt_tuning_config = cls.create_hf_tuning_config(
        base_model=base_model,
        tuning_type=tuning_type,
        task_type=task_type,
        tokenizer_name_or_path=tokenizer_name_or_path,
        tuning_config=tuning_config,
        output_model_types=output_model_types,
    )
    return prompt_tuning_config


def _create_lora_config(tuning_config, task_type) -> LoraConfig:
    """Creates Huggingface LoraConfig from Caikit tuning configuration."""

    config_kwargs = tuning_config.to_dict()
    log.info("<NLP61012781I>", f"Parameters used: {config_kwargs}")
    config_params = _filter_params_for_prompt_config(tuning_config, config_kwargs)
    if "output_model_types" in config_params:
        del config_params["output_model_types"]
    lora_config = LoraConfig(task_type=task_type, **config_params)
    return lora_config
