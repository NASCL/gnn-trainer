
import os
import logging
import json
from enum import Enum
from dataclasses import asdict, dataclass, field
from typing import Optional
from functools import cached_property
import math

import torch

from optimization import SchedulerType
from trainer_utils import IntervalStrategy


logger = logging.getLogger(__name__)


def default_logdir() -> str:
    """
    Same default as PyTorch
    """
    import socket
    from datetime import datetime

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    # return os.path.join('runs', f'{current_time}_{socket.gethostname()}')
    return os.path.join('runs', current_time)


@dataclass
class TrainingArguments:

    output_dir: str = field(
        metadata={'help': 'The output directory where model predictions and checkpoints will be written.'},
    )

    do_train: bool = field(default=False, metadata={'help': 'Whether to run training.'})
    do_eval: bool = field(default=False, metadata={'help': 'Whether to run eval on the validation set.'})
    do_predict: bool = field(default=False, metadata={'help': 'Whether to run eval on the test set.'})

    do_early_stopping: bool = field(
        default=False, metadata={'help': 'Whether to perform early stopping during training.'}
    )
    early_stopping_patience: int = field(
        default=None, metadata={'help': 'Stop training when the specified metric worsens for X epochs.'}
    )

    trainable_loss: bool = field(
        default=False, metadata={'help': 'Whether the loss module has trainable parameters to optimize.'}
    )

    evaluation_strategy: IntervalStrategy = field(
        default='no',
        metadata={'help': 'The evaluation strategy to use.'}
    )
    eval_steps: int = field(default=None, metadata={'help': 'Run an evaluation every X steps.'})

    prediction_loss_only: bool = field(
        default=False,
        metadata={'help': 'When performing evaluation and prediction, return only the loss.'}
    )

    train_batch_size: int = field(
        default=8, metadata={'help': 'Batch size for training.'}
    )
    eval_batch_size: int = field(
        default=8, metadata={'help': 'Batch size for evaluation.'}
    )

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={'help': 'Number of update steps to accumulate before performing a backward/update pass.'}
    )

    learning_rate: float = field(default=5e-4, metadata={'help': 'The initial learning rate for AdamW.'})
    weight_decay: float = field(default=0.0, metadata={'help': 'Weight decay for AdamW if applied.'})
    adam_beta1: float = field(default=0.9, metadata={'help': 'Beta1 for AdamW optimizer.'})
    adam_beta2: float = field(default=0.999, metadata={'help': 'Beta2 for AdamW optimizer.'})
    adam_epsilon: float = field(default=1e-8, metadata={'help': 'Epsilon for AdamW optimizer.'})
    max_grad_norm: float = field(default=1.0, metadata={'help': 'Max gradient norm.'})

    num_train_epochs: float = field(default=50.0, metadata={'help': 'Total number of training epochs to perform.'})
    lr_scheduler_type: SchedulerType = field(default='linear', metadata={'help': 'The scheduler type to use.'})
    warmup_ratio: float = field(
        default=0.0,
        metadata={'help': 'Linear warmup over warmup_ratio fraction of total steps.'}
    )
    warmup_steps: int = field(default=0, metadata={'help': 'Linear warmup of warmup_steps.'})

    log_level: Optional[str] = field(
        default='info',
        metadata={'help': 'Logger log level to use. Possible choices are the log levels as strings:'
                          '"debug", "info", "warning", "error", and "critical". Defaults to "info".'}
    )
    logging_strategy: IntervalStrategy = field(default='steps', metadata={'help': 'The logging strategy to use.'})
    logging_first_step: bool = field(default=False, metadata={'help': 'Log the first global_step.'})
    logging_steps: int = field(default=500, metadata={'help': 'Log every X update steps.'})
    logging_nan_inf_filter: str = field(default=True, metadata={'help': 'Filter nan and inf losses for logging.'})

    save_strategy: IntervalStrategy = field(
        default='epoch',
        metadata={'help': 'The checkpoint save strategy to use.'}
    )
    save_steps: int = field(default=500, metadata={'help': 'Save checkpoint every X update steps.'})
    save_total_limit: Optional[int] = field(
        default=None,
        metadata={'help': 'Limit the total number of checkpoints. '
                          'Deletes the older checkpoints in the output_dir. Default is unlimited checkpoints.'}
    )

    no_cuda: bool = field(default=False, metadata={'help': 'Do not use CUDA even when it is available.'})
    seed: int = field(default=42, metadata={'help': 'Random seed that will be set a the beginning of training.'})
    dataloader_drop_last: bool = field(
        default=False, metadata={'help': 'Drop the last incomplete batch if it is not divisible by the batch size.'}
    )
    dataloader_num_workers: int = field(
        default=0,
        metadata={'help': 'Number of subprocesses to use for data loading. '
                          '0 means that the data will be loaded in the main process.'}
    )
    dataloader_pin_memory: bool = field(
        default=False, metadata={'help': 'Whether or not to pin memory for DataLoader.'}
    )
    disable_tqdm: Optional[bool] = field(
        default=None, metadata={'help': 'Whether or not to disable the tqdm progress bars.'}
    )

    load_best_model_at_end: Optional[bool] = field(
        default=False,
        metadata={'help': 'Whether or not to load the best model found during training and the end of training.'}
    )
    metric_for_best_model: Optional[str] = field(
        default=None, metadata={'help': 'The metric to use to compare two different models.'}
    )
    greater_is_better: Optional[bool] = field(
        default=None, metadata={'help': 'Whether the `metric_for_best_model` should be maximized or not.'}
    )

    def __post_init__(self):
        self.log_level = logging.getLevelName(self.log_level.upper())

        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)
            self.logging_dir = os.path.expanduser(
                os.path.join(self.output_dir, default_logdir())
            )
            os.makedirs(self.logging_dir, exist_ok=True)

        if self.disable_tqdm is None:
            self.disable_tqdm = logger.getEffectiveLevel() > logging.WARNING

        self.evaluation_strategy = IntervalStrategy(self.evaluation_strategy)
        self.logging_strategy = IntervalStrategy(self.logging_strategy)
        self.save_strategy = IntervalStrategy(self.save_strategy)

        self.lr_scheduler_type = SchedulerType(self.lr_scheduler_type)
        if not self.do_eval and self.evaluation_strategy != IntervalStrategy.NO:
            self.do_eval = True

        # logging_steps must be non-zero for logging_strategy other than 'no'.
        if self.logging_strategy == IntervalStrategy.STEPS and self.logging_steps == 0:
            raise ValueError(f'Logging strategy {self.logging_strategy} requires non-zero `logging_steps`.')

        # Sanity checks for load_best_model_at_end: We require save and eval strategies to be compatible.
        if self.do_early_stopping:
            if self.early_stopping_patience is None:
                raise ValueError('Early stopping enabled, but no patience limit is set.')

        if self.load_best_model_at_end:
            if self.evaluation_strategy != self.save_strategy:
                raise ValueError(
                    f'`load_best_model_at_end` requires the save and eval strategy to match, but found\n'
                    f'- Evaluation strategy: {self.evaluation_strategy}\n'
                    f'- Save strategy: {self.save_strategy}'
                )

        if self.load_best_model_at_end and self.metric_for_best_model is None:
            self.metric_for_best_model = 'loss'
        if self.greater_is_better is None and self.metric_for_best_model is not None:
            self.greater_is_better = self.metric_for_best_model not in ['loss', 'eval_loss']

        if self.warmup_ratio < 0 or self.warmup_ratio > 1:
            raise ValueError('warmup_ratio must lie in the range [0, 1].')
        elif self.warmup_ratio > 0 and self.warmup_steps > 0:
            logger.info(
                f'Both warmup_ratio and warmup_steps given, '
                f'warmup_steps will override any effects of warmup_ratio during training.'
            )

    def __str__(self):
        self_as_dict = asdict(self)

        attrs_as_str = [f'{k}={v},\n' for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    __repr__ = __str__

    @cached_property
    def _setup_devices(self) -> 'torch.device':
        logger.info('PyTorch: setting up devices.')
        if self.no_cuda:
            device = 'cpu'
        else:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        if device.type == 'cuda':
            torch.cuda.set_device(device)

        return device

    @property
    def device(self) -> 'torch.device':
        return self._setup_devices

    def get_log_level(self):
        return self.log_level

    def get_warmup_steps(self, num_training_steps: int):
        warmup_steps = (
            self.warmup_steps if self.warmup_steps > 0 else math.ceil(num_training_steps * self.warmup_ratio)
        )
        return warmup_steps

    def to_dict(self):
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
        return d

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2)
