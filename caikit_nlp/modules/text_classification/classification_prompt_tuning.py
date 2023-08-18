# Standard
from typing import Any, Dict, List

# Third party
from transformers.models.auto.tokenization_auto import AutoTokenizer
from peft import (
    PeftModel,
)

# First party
from caikit.core.data_model import DataStream
from caikit.core.modules import ModuleBase, ModuleSaver, module

# Local
from ...data_model import (
    ClassificationTrainRecord,
    PromptOutputModelType,
    TuningConfig,
)
from .text_classification_task import TextClassificationTask
from ..text_generation import PeftPromptTuning
from ..text_generation.peft_prompt_tuning import TuningType
from ...toolkit.task_specific_utils import get_sorted_unique_class_labels

# TODO: try to refactor this into a smaller module
# pylint: disable=too-many-lines,too-many-instance-attributes
@module(
    id="6713731b-160b-4sc5-8df4-167126e2cd11",
    name="Classification Peft Tuning",
    version="0.1.0",
    task=TextClassificationTask,
)

class ClassificationPeftPromptTuning(ModuleBase):

    _DETECT_DEVICE = "__DETECT__"

    def __init__(
        self,
        classifier: PeftPromptTuning,
        unique_class_labels: List[str],
    ):
        
        self.classifier = classifier
        self.unique_class_labels = unique_class_labels

    def __del__(self):
        del self.model
        del self.tokenizer
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except AttributeError:
            pass
    
    @classmethod
    def train(
        cls,
        base_model: str,  # TODO: Union[str, PretrainedModelBase]
        train_stream: DataStream[ClassificationTrainRecord],
        tuning_config: TuningConfig,
        val_stream: DataStream[ClassificationTrainRecord] = None,
        device: str = _DETECT_DEVICE,  # TODO: Union[int, str]
        tuning_type: str = "PROMPT_TUNING",  # TODO: Union[str, TuningType]
        num_epochs: int = 20,
        lr: float = 0.3,
        verbalizer: str = "{{input}}",
        batch_size: int = 8,
        max_source_length: int = 256,
        max_target_length: int = 128,
        accumulate_steps: int = 32,
        torch_dtype: str = None,  # TODO: Optional[Union[torch.dtype, str]]
        silence_progress_bars: bool = True,
        **kwargs,
    ) -> "ClassificationPeftPromptTuning":
        """Run prompt tuning (vanilla or MPT) through PEFT on a CausalLM or Seq2seq model
        to refine a text generation model.

        Args:
            base_model:  Union[str, caikit_nlp.resources.pretrained_model.base.PretrainedModelBase]
                Base resource model used for underlying generation.
            train_stream: DataStream[GenerationTrainRecord] or DataStream[ClassificationTrainRecord]
                Data to be used for training the prompt vectors of the generation model.
            tuning_config: TuningConfig
                Additional model tuning configurations to be considered for prompt vector
                initialization and training behavior.
            val_stream: Optional[DataStream[GenerationTrainRecord]
                           or DataStream[ClassificationTrainRecord]]
                Data to be used for validation throughout the train process or None.
            device: str
                Device to be used for training the model. Default: cls._DETECT_DEVICE, which
                will fall back to "cuda" if available, else None.
            tuning_type: str
                Type of Peft Tuning config which we would like to build.
            num_epochs: int
                Number of epochs to tune the prompt vectors. Default: 20.
            lr: float
                Learning rate to be used while tuning prompt vectors. Default: 1e-3.
            verbalizer: str
                Verbalizer template to be used for formatting data at train and inference time.
                This template may use brackets to indicate where fields from the data model
                TrainGenerationRecord must be rendered. Default: "{{input}}", i.e., the raw text.
            batch_size: int
                Batch sized to be used for training / evaluation data. Default: 8.
            max_source_length: int
                Max length of input sequences being considered. Default: 256.
            max_target_length: int
                Max length of target sequences being predicted. Default: 128.
            accumulate_steps: int
                Number of steps to use for gradient accumulation. Default: 1.
            torch_dtype: str
                TODO: Optional[Union[torch.dtype, str]]
                Data type to use for training/inference of the underlying text generation model.
                If no value is provided, we pull from torch_dtype in config. If an in memory
                resource is provided which does not match the specified data type, the model
                underpinning the resource will be converted in place to the correct torch dtype.
            silence_progress_bars: bool
                Silences TQDM progress bars at train time. Default: True.
        Returns:
            PeftPromptTuning
                Instance of this class with tuned prompt vectors.
        """

        
        unique_class_labels = get_sorted_unique_class_labels(train_stream)
        # Wrap up the trained model in a class instance
        return cls(
            classifier = PeftPromptTuning.train(base_model, train_stream, tuning_config,
                               val_stream,  device, tuning_type, num_epochs, 
                               lr, verbalizer, batch_size, max_source_length, 
                               max_target_length, accumulate_steps, torch_dtype,
                               silence_progress_bars, **kwargs),
            unique_class_labels=unique_class_labels,
            # TODO: Export other training params to model as well
        )

    def save(self, model_path, save_base_model: bool = False):
        """Save classification model

        Args:
            model_path: str
                Folder to save classification prompt tuning model
            save_base_model: bool
                Save base model along with the prompts in the model_path provided.
                Default: False
        """
        saver = ModuleSaver(self, model_path=model_path)
        classifier_artifacts_path = "classifier"
        self.classifier.save(classifier_artifacts_path, save_base_model)
        with saver:
            saver.update_config(
                {
                     "unique_class_labels": {self.unique_class_labels},
                     "classifier_path": classifier_artifacts_path
                }
            )
        