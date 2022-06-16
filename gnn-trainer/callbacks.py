
import json
import logging
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict

import numpy as np
from tqdm.auto import tqdm

from training_args import TrainingArguments
from trainer_utils import IntervalStrategy


"""
Trainer Callbacks adapted from HuggingFace:
https://github.com/huggingface/transformers/blob/v4.15.0/src/transformers/trainer_callback.py
"""

logger = logging.getLogger(__name__)


@dataclass
class TrainerState:
    epoch: Optional[float] = None
    global_step: int = 0
    max_steps: int = 0
    num_train_epochs: int = 0
    log_history: List[Dict[str, float]] = None
    best_metric: Optional[float] = None
    best_model_checkpoint: Optional[str] = None

    def __post_init__(self):
        if self.log_history is None:
            self.log_history = []

    def save_to_json(self, json_path: str):
        json_string = json.dumps(asdict(self), indent=2, sort_keys=True) + '\n'
        with open(json_path, 'w', encoding='utf-8')as f:
            f.write(json_string)

    @classmethod
    def load_from_json(cls, json_path:str):
        with open(json_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return cls(**json.loads(text))


@dataclass
class TrainerControl:
    should_training_stop: bool = False
    should_epoch_stop: bool = False
    should_save: bool = False
    should_evaluate: bool = False
    should_log: bool = False

    def _new_training(self):
        """Internal method that resets the variable for a new training."""
        self.should_training_stop = False

    def _new_epoch(self):
        """Internal method that resets the variable for a new epoch."""
        self.should_epoch_stop = False

    def _new_step(self):
        """Internal method that resets the variable for a new step."""
        self.should_save = False
        self.should_evaluate = False
        self.should_log = False


class TrainerCallback:
    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of the initialization of the [`Trainer`].
        """
        pass

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of training.
        """
        pass

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of training.
        """
        pass

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of an epoch.
        """
        pass

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of an epoch.
        """
        pass

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        pass

    def on_substep_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of a substep during gradient accumulation.
        """
        pass

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        pass

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after an evaluation phase.
        """
        pass

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after a checkpoint save.
        """
        pass

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after logging the last logs.
        """
        pass

    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after a prediction step.
        """
        pass


class CallbackHandler(TrainerCallback):
    """Internal class that just calls the list of callbacks in order"""
    def __init__(self, callbacks, model, optimizer, lr_scheduler):
        self.callbacks = []
        for cb in callbacks:
            self.add_callback(cb)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.train_dataloader = None
        self.eval_dataloader = None

        if not any(isinstance(cb, DefaultFlowCallback) for cb in self.callbacks):
            logger.warning(
                'The Trainer will not work properly without a `DefaultFlowCallback` in its callbacks.\n'
                'You should add one before training with `trainer.add_callback(DefaultFlowCallback).\n'
                f'The current list of callbacks is:\n'
                f'{self.callback_list}'
            )

    def add_callback(self, callback):
        cb = callback() if isinstance(callback, type) else callback
        cb_class = callback if isinstance(callback, type) else callback.__class__
        if cb_class in [c.__class__ for c in self.callbacks]:
            logger.warning(
                f'You are adding a {cb_class} to the callbacks of this Trainer, but there is already one.\n'
                f'The current list of callbacks is:\n'
                f'{self.callback_list}'
            )
        self.callbacks.append(cb)

    def pop_callback(self, callback):
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return cb
        else:
            for cb in self.callbacks:
                if cb == callback:
                    self.callbacks.remove(cb)
                    return cb

    def remove_callback(self, callback):
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return
        else:
            self.callbacks.remove(callback)

    @property
    def callback_list(self):
        return '\n'.join(cb.__class__.__name__ for cb in self.callbacks)

    def call_event(self, event: str, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        for callback in self.callbacks:
            result = getattr(callback, event)(
                args,
                state,
                control,
                model=self.model,
                optimizer=self.optimizer,
                lr_scheduler=self.scheduler,
                train_dataloader=self.train_dataloader,
                eval_dataloader=self.eval_dataloader,
                **kwargs,
            )
            # A Callback can skip the return of `control` if it doesn't change it.
            if result is not None:
                control = result
        return control

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        return self.call_event('on_init_end', args, state, control)

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        control.should_training_stop = False
        return self.call_event('on_train_begin', args, state, control)

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        return self.call_event('on_train_end', args, state, control)

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        control.should_epoch_stop = False
        return self.call_event('on_epoch_begin', args, state, control)

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        return self.call_event('on_epoch_end', args, state, control)

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        control.should_log = False
        control.should_evaluate = False
        control.should_save = False
        return self.call_event('on_step_begin', args, state, control)

    def on_substep_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        return self.call_event('on_substep_end', args, state, control)

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        return self.call_event('on_step_end', args, state, control)

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        control.should_evaluate = False
        return self.call_event('on_evaluate', args, state, control, **kwargs)

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        control.should_save = False
        return self.call_event('on_save', args, state, control)

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        control.should_log = False
        return self.call_event('on_log', args, state, control, **kwargs)

    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        return self.call_event('on_prediction_step', args, state, control)


class DefaultFlowCallback(TrainerCallback):
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Log
        if args.logging_strategy == IntervalStrategy.EPOCH:
            control.should_log = True

        # Evaluate
        if args.evaluation_strategy == IntervalStrategy.EPOCH:
            control.should_evaluate = True

        # Save
        if args.save_strategy == IntervalStrategy.EPOCH:
            control.should_save = True

        return control


class ProgressCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that displays the progress of training or evaluation.
    """

    def __init__(self):
        self.training_bar = None
        self.prediction_bar = None
        self.current_step = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.training_bar = tqdm(total=state.max_steps)
        self.current_step = 0

    def on_step_end(self, args, state, control, loss=None, **kwargs):
        self.training_bar.update(state.global_step - self.current_step)
        self.current_step = state.global_step
        if loss:
            self.training_bar.set_postfix({
                'loss': f'{loss:.4f}'
            })

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        if self.prediction_bar is None:
            self.prediction_bar = tqdm(total=len(eval_dataloader), leave=self.training_bar is None)
        self.prediction_bar.update(1)

    def on_evaluate(self, args, state, control, **kwargs):
        if self.prediction_bar is not None:
            self.prediction_bar.close()
        self.prediction_bar = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        pass

    def on_train_end(self, args, state, control, **kwargs):
        self.training_bar.close()
        self.training_bar = None


class PrinterCallback(TrainerCallback):
    """
    A bare [`TrainerCallback`] that just prints the logs.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        print(logs)


class EarlyStoppingCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that handles early stopping.
    Args:
        early_stopping_patience: (`int`):
            Use with `metric_for_best_model` to stop training when the specified metric worsens for
            `early_stopping_patience` evaluation calls.
    This callback depends on [`TrainingArguments`] argument *load_best_model_at_end* functionality
    to set best_metric in [`TrainerState`].
    """

    def __init__(self, early_stopping_patience: int = 5):
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_patience_counter = 0

    def check_metric_value(self, args, state, metric_value):
        operator = np.greater if args.greater_is_better else np.less
        if state.best_metric is None or operator(metric_value, state.best_metric):
            self.early_stopping_patience_counter = 0
        else:
            self.early_stopping_patience_counter += 1

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        assert args.load_best_model_at_end, 'Early stopping requires load_best_model_at_end = True.'
        assert args.metric_for_best_model is not None, 'Early stopping requires metric_for_best_model is defined.'
        assert args.evaluation_strategy != IntervalStrategy.NO, 'Early stopping requires IntervalStrategy of steps or epochs'

    def on_evaluate(
            self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics: dict = None, **kwargs
    ):
        metric_to_check = args.metric_for_best_model
        if not metric_to_check.startswith('eval_'):
            metric_to_check = f'eval_{metric_to_check}'
        metric_value = metrics.get(metric_to_check)

        if metric_value is None:
            logger.warning(
                f'Early stopping requires metric_for_best_model, '
                f'but did not find {metric_to_check} so early stopping is disabled.'
            )
            return

        self.check_metric_value(args, state, metric_value)
        if self.early_stopping_patience_counter > self.early_stopping_patience:
            control.should_training_stop = True
