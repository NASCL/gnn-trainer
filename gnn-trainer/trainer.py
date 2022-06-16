
import math
import os
import time
import logging
import warnings
from typing import Optional, Callable, List, Tuple, Dict

import numpy as np
import torch
from torch import nn, optim

import dgl
from dgl import DGLGraph
from dgl.dataloading import GraphDataLoader, GraphCollator

# TODO: sklearn metrics functions
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.exceptions import UndefinedMetricWarning

from training_args import TrainingArguments

from optimization import get_scheduler

from callbacks import (
    TrainerState,
    TrainerControl,
    TrainerCallback,
    CallbackHandler,
    DefaultFlowCallback,
    EarlyStoppingCallback,
    PrinterCallback,
    ProgressCallback,
)

from trainer_utils import set_seed, get_parameter_names, speed_metrics

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

# Names of files used for checkpointing
CONFIG_NAME = 'config.json'
WEIGHTS_NAME = 'pytorch_model.bin'
TRAINING_ARGS_NAME = 'training_args.bin'
TRAINER_STATE_NAME = 'trainer_state.json'
OPTIMIZER_NAME = 'optimizer.pt'
SCHEDULER_NAME = 'scheduler.pt'
PREFIX_CHECKPOINT_DIR = 'checkpoint'

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

# TODO: Saving model, optimizer, scheduler states
# TODO: Checkpoint limit. Delete older checkpoints
# TODO: Resuming training from checkpoint
# TODO: Half-precision support
# TODO: Distributed data parallel support
"""
https://github.com/huggingface/transformers/tree/v4.15.0/src/transformers
https://github.com/huggingface/transformers/blob/v4.15.0/src/transformers/trainer.py
"""


class Trainer:
    def __init__(
            self,
            model: nn.Module = None,
            loss_module: nn.Module = None,
            args: TrainingArguments = None,
            collate_fn: Optional[Callable] = None,
            train_dataset: Optional[Tuple[List[DGLGraph], torch.Tensor]] = (None, None),
            eval_dataset: Optional[Tuple[List[DGLGraph], torch.Tensor]] = (None, None),
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[optim.Optimizer, optim.lr_scheduler.LambdaLR] = (None, None),
    ):
        if args is None:
            output_dir = 'tmp_trainer'
            logger.info(f'No `TrainingArguments` passed, using `output_dir={output_dir}.')
            args = TrainingArguments(output_dir=output_dir)
        self.args = args

        set_seed(self.args.seed)
        self.is_in_train = False

        log_level = args.get_log_level()
        logger.setLevel(log_level)

        if model is None:
            raise RuntimeError('`Trainer` requires a `model`.')

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        default_collate_fn = GraphCollator
        self.collate_fn = collate_fn if collate_fn is not None else default_collate_fn

        self.model = self._move_model_to_device(model, args.device)

        if loss_module is not None:
            loss_module = self._move_model_to_device(loss_module, args.device)
        self.loss_module = loss_module

        self.optimizer, self.lr_scheduler = optimizers

        default_callbacks = DEFAULT_CALLBACKS
        callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            callbacks, self.model, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)
        if args.do_early_stopping:
            self.add_callback(EarlyStoppingCallback)

        self._loggers_initialized = False

        os.makedirs(self.args.output_dir, exist_ok=True)

        if self.collate_fn is not None and not callable(getattr(self.collate_fn, 'collate', None)):
            raise ValueError('The `collate_fn` should have a callable "collate" function.')

        self.state = TrainerState()
        self.control = TrainerControl()

    @staticmethod
    def _move_model_to_device(model, device):
        return model.to(device)

    @staticmethod
    def num_examples(dataloader):
        if dgl.__version__.startswith('0.7'):
            return len(dataloader.dataloader.dataset)
        else:
            return len(dataloader.dataset)

    def add_callback(self, callback):
        self.callback_handler.add_callback(callback)

    def pop_callback(self, callback):
        return self.callback_handler.pop_callback(callback)

    def remove_callback(self, callback):
        self.callback_handler.remove_callback(callback)

    def get_train_dataloader(self) -> GraphDataLoader:
        if self.train_dataset is None:
            raise ValueError('Trainer: training requires a train_dataset.')

        train_dataset = self.train_dataset

        return GraphDataLoader(
            dataset=train_dataset,
            batch_size=self.args.train_batch_size,
            collate_fn=None,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=self.args.dataloader_drop_last,
            shuffle=True,
        )

    def get_eval_dataloader(
            self, eval_dataset: Optional[Tuple[List[DGLGraph], torch.Tensor]] = None
    ) -> GraphDataLoader:

        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError('Trainer: evaluation requires an eval_dataset.')
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        return GraphDataLoader(
            dataset=eval_dataset,
            batch_size=self.args.eval_batch_size,
            collate_fn=None,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=self.args.dataloader_drop_last,
            shuffle=False,
        )

    def get_test_dataloader(self, test_dataset: Tuple[List[DGLGraph], torch.Tensor] = None) -> GraphDataLoader:
        return GraphDataLoader(
            dataset=test_dataset,
            batch_size=self.args.eval_batch_size,
            collate_fn=None,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=self.args.dataloader_drop_last,
            shuffle=False,
        )

    def create_optimizer(self):
        if self.optimizer is None:
            optimizer_kwargs = {
                'betas': (self.args.adam_beta1, self.args.adam_beta2),
                'eps': self.args.adam_epsilon,
                'lr': self.args.learning_rate
            }

            model_decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
            model_decay_parameters = [name for name in model_decay_parameters if 'bias' not in name]
            optimizer_grouped_parameters = [
                {
                    'params': [p for n, p in self.model.named_parameters() if n in model_decay_parameters],
                    'weight_decay': self.args.weight_decay,
                },
                {
                    'params': [p for n, p in self.model.named_parameters() if n not in model_decay_parameters],
                    'weight_decay': 0.0,
                },
            ]

            if self.args.trainable_loss:
                loss_decay_parameters = get_parameter_names(self.loss_module, [nn.LayerNorm])
                loss_decay_parameters = [name for name in loss_decay_parameters if 'bias' not in name]

                optimizer_grouped_parameters.append({
                    'params': [p for n, p in self.loss_module.named_parameters() if n in loss_decay_parameters],
                    'weight_decay': self.args.weight_decay,
                })
                optimizer_grouped_parameters.append({
                    'params': [p for n, p in self.loss_module.named_parameters() if n not in loss_decay_parameters],
                    'weight_decay': 0.0,
                })

            self.optimizer = optim.AdamW(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        self.create_optimizer()
        self.create_scheduler(num_training_steps=num_training_steps, optimizer=self.optimizer)

    def create_scheduler(self, num_training_steps: int, optimizer: optim.Optimizer = None):
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )
        return self.lr_scheduler

    def train(
            self,
            **kwargs
    ):
        args = self.args

        self.is_in_train = True

        self.model = self._move_model_to_device(self.model, args.device)
        if self.loss_module is not None:
            self.loss_module = self._move_model_to_device(self.loss_module, args.device)

        train_dataloader = self.get_train_dataloader()

        # Set up training control variables
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(len(train_dataloader) // args.gradient_accumulation_steps, 1)

        max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
        num_train_epochs = math.ceil(args.num_train_epochs)
        num_train_samples = len(self.train_dataset) * args.num_train_epochs

        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()

        model = self._move_model_to_device(self.model, args.device)
        loss_module = None if self.loss_module is None else self._move_model_to_device(self.loss_module, args.device)

        # Train
        num_examples = self.num_examples(train_dataloader)

        logger.info('***** Running Training *****')
        logger.info(f'  Num examples = {num_examples}')
        logger.info(f'  Num epochs = {num_train_epochs}')
        logger.info(f'  Batch size = {args.train_batch_size}')
        logger.info(f'  Gradient accumulation steps = {args.gradient_accumulation_steps}')
        logger.info(f'  Total optimization steps = {max_steps}')

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0

        # Update references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        # These should be the same if the state has been saved but in case the training arguments changes,
        # it's safer to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs

        tr_loss = torch.tensor(0.0, device=args.device)
        self._total_loss_scalar = 0.0
        self._global_step_last_logged = self.state.global_step

        self.optimizer.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        metrics = {}
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader

            steps_in_epoch = len(epoch_iterator)

            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                tr_loss_step = self.training_step(model, loss_module, inputs)

                if (
                        args.logging_nan_inf_filter
                        and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # If loss is nan or inf, simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._global_step_last_logged)
                else:
                    tr_loss += tr_loss_step

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                        # Last step in the epoch, but step is always smaller than gradient_accumulation_steps
                        args.gradient_accumulation_steps >= steps_in_epoch == (step + 1)
                ):
                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        if hasattr(self.optimizer, 'clip_grad_norm'):
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, 'clip_grad_norm_'):
                            model.clip_grad_norm_(args.max_grad_norm)
                        else:
                            nn.utils.clip_grad_norm_(
                                model.parameters(),
                                args.max_grad_norm
                            )

                    # Optimizer step
                    self.optimizer.step()
                    self.lr_scheduler.step()

                    self.optimizer.zero_grad()

                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(
                        args, self.state, self.control, loss=tr_loss.item()
                    )

                    tmp_metrics = self._maybe_log_save_evaluate(tr_loss, model, epoch)
                    if tmp_metrics:
                        metrics[f'epoch_{epoch:02d}_{step:03d}'] = tmp_metrics

                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            if step < 0:
                logger.warning(
                    f'There seems to be not a single sample in your epoch_iterator, stopping training at step '
                    f'{self.state.global_step}! This is expected if you\'re using an IterableDataset and set '
                    f'num_steps ({max_steps}) higher than the number of available samples.'
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            tmp_metrics = self._maybe_log_save_evaluate(tr_loss, model, epoch)
            if tmp_metrics:
                metrics[f'epoch_{epoch:02d}'] = tmp_metrics

            if self.control.should_training_stop:
                break

        logger.info(f'\n\nTraining completed.')
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            logger.info(
                f'Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).'
            )

            best_model_path = os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME)
            if os.path.exists(best_model_path):
                state_dict = torch.load(best_model_path, map_location='cpu')
                self._load_state_dict_in_model(state_dict)
            else:
                logger.warning(f'Could not locate the best model at {best_model_path}')

        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        # metrics = speed_metrics('train', start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        metrics['train_loss'] = train_loss

        self.is_in_train = False

        self.log(metrics)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return self.state.global_step, train_loss, metrics

    def training_step(self, model: nn.Module, loss_module: nn.Module, inputs: Tuple[DGLGraph, torch.Tensor]):
        model.train()
        inputs = self._prepare_inputs(inputs)

        loss = self.compute_loss(model, loss_module, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss /= self.args.gradient_accumulation_steps

        loss.backward()

        return loss.detach()

    def compute_loss(
            self,
            model: nn.Module,
            loss_module: nn.Module,
            inputs: Tuple[DGLGraph, torch.Tensor],
            return_outputs: bool = False
    ):
        batched_graph, labels = inputs
        outputs = model(batched_graph, batched_graph.ndata.pop('feat').float())

        outputs = outputs.squeeze()
        labels = labels.squeeze()

        loss = loss_module(outputs, labels)

        return (loss, outputs) if return_outputs else loss

    def _prepare_inputs(self, inputs: Tuple[DGLGraph, torch.Tensor]):
        batched_graph, labels = inputs
        return batched_graph.to(self.args.device), labels.squeeze().to(self.args.device)

    def _load_state_dict_in_model(self, state_dict):
        load_result = self.model.load_state_dict(state_dict, strict=False)

        if len(load_result.missing_keys) != 0:
            pass

    def load_model(self, checkpoint_dir: str, model=None):
        if self.model is None and model is None:
            raise ValueError('No model provided.')
        elif model:
            self.model = model

        if not os.path.isfile(os.path.join(checkpoint_dir, WEIGHTS_NAME)):
            raise ValueError(f'Cannot find a valid checkpoint at {checkpoint_dir}')

        state_dict = torch.load(os.path.join(checkpoint_dir, WEIGHTS_NAME), map_location='cpu')
        self._load_state_dict_in_model(state_dict)
        del state_dict  # Free memory

    def log(self, logs: Dict[str, float]) -> None:
        if self.state.epoch is not None:
            logs['epoch'] = round(self.state.epoch, 2)

        output = {**logs, **{'step': self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs=logs)

    def _maybe_log_save_evaluate(self, tr_loss, model, epoch):
        if self.control.should_log:
            logs: Dict[str, float] = {}

            tr_loss_scalar = tr_loss.mean().item()
            tr_loss -= tr_loss

            logs['loss'] = round(tr_loss_scalar / (self.state.global_step - self._global_step_last_logged), 4)
            logs['learning_rate'] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._global_step_last_logged = self.state.global_step

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate()
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics=metrics)

        if self.control.should_save:
            self._save_checkpoint(model, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

        return metrics

    def _get_learning_rate(self):
        return self.lr_scheduler.get_last_lr()[0]

    def _save_checkpoint(self, model, metrics=None):
        checkpoint_folder = f'{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}'
        output_dir = os.path.join(self.args.output_dir, checkpoint_folder)
        os.makedirs(output_dir, exist_ok=True)

        self.save_model(output_dir)

        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith('eval_'):
                metric_to_check = f'eval_{metric_to_check}'
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                    self.state.best_metric is None
                    or self.state.best_model_checkpoint is None
                    or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

    def save_model(self, output_dir: Optional[str] = None):
        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        torch.save(self.model.state_dict(), os.path.join(output_dir, WEIGHTS_NAME))
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def evaluate(
            self,
            eval_dataset: Optional[Tuple[List[DGLGraph], torch.Tensor]] = None,
            metric_key_prefix: str = 'eval',
    ) -> Dict[str, float]:

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        num_samples = self.num_examples(eval_dataloader)

        start_time = time.time()
        metrics = self.evaluation_loop(
            eval_dataloader,
            description='Evaluation',
            prediction_loss_only=None,
            metric_key_prefix=metric_key_prefix
        )

        # metrics.update(
        #     speed_metrics(
        #         metric_key_prefix,
        #         start_time,
        #         num_samples=num_samples,
        #         num_steps=math.ceil(num_samples / self.args.eval_batch_size),
        #     )
        # )

        self.log(metrics)

        return metrics

    # TODO: Temp hard-code evaluation metrics
    def compute_metrics(self, preds: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5):
        is_multiclass = labels.max().item() > 1
        if is_multiclass:
            preds = torch.argmax(preds, dim=-1)
            probs = preds.tolist()  # Predicted class not raw probs
        else:
            probs = preds.tolist()
            preds = (preds > threshold).float()

        return {
            'accuracy': accuracy_score(preds, labels),
            'precision': precision_score(preds, labels, average='micro' if is_multiclass else 'binary'),
            'recall': recall_score(preds, labels, average='micro' if is_multiclass else 'binary'),
            'F1 micro': f1_score(preds, labels, average='micro'),
            'F1 macro': f1_score(preds, labels, average='macro'),
            'probs': probs,
            'labels': labels.tolist(),
        }

    def evaluation_loop(
            self,
            dataloader: GraphDataLoader,
            description: str,
            prediction_loss_only: Optional[bool] = None,
            metric_key_prefix: str = 'eval',
    ):
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        logger.info(f'***** Running {description} *****')
        logger.info(f'  Num examples = {self.num_examples(dataloader)}')
        logger.info(f'  Batch size = {args.eval_batch_size}')

        model = self.model
        model.eval()

        self.callback_handler.eval_dataloader = dataloader

        all_losses, all_preds, all_labels = None, None, None
        for step, inputs in enumerate(dataloader):
            losses, logits, labels = self.prediction_step(model, inputs, prediction_loss_only)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            losses = losses.unsqueeze(-1)

            try:
                all_losses = losses if all_losses is None else torch.cat((all_losses, losses), dim=0)
                all_preds = logits if all_preds is None else torch.cat((all_preds, logits), dim=0)
                all_labels = labels if all_labels is None else torch.cat((all_labels, labels), dim=0)
            except RuntimeError:
                print(logits.shape)
                print(all_preds if all_preds is None else all_preds.shape)

        all_losses = all_losses.cpu()
        all_preds = all_preds.cpu()
        all_labels = all_labels.cpu()

        metrics = self.compute_metrics(all_preds, all_labels)

        # TODO: Remove tensor/np types for JSON serialization

        if all_losses is not None:
            metrics[f'{metric_key_prefix}_loss'] = all_losses.mean().item()

        metrics_with_prefix = {}
        for key in metrics:
            val = metrics[key]
            if not key.startswith(f'{metric_key_prefix}_'):
                metrics_with_prefix[f'{metric_key_prefix}_{key}'] = val
            else:
                metrics_with_prefix[key] = val

        return metrics_with_prefix

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Tuple[DGLGraph, torch.Tensor],
            prediction_loss_only: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = self._prepare_inputs(inputs)

        labels = inputs[1].detach()

        with torch.no_grad():
            loss, logits = self.compute_loss(model, self.loss_module, inputs, return_outputs=True)
        loss = loss.mean().detach()

        if prediction_loss_only:
            return loss, None, None

        preds = torch.sigmoid(logits.detach())
        # _, preds = logits.detach().max(dim=-1)

        return loss, preds, labels
