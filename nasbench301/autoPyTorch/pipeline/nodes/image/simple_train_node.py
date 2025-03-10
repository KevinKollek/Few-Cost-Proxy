__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

import torch
import time
import logging

import os, pprint
import scipy.sparse
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import copy
import json

from autoPyTorch.pipeline.base.pipeline_node import PipelineNode

import ConfigSpace
import ConfigSpace.hyperparameters as CSH
from autoPyTorch.utils.configspace_wrapper import ConfigWrapper
from autoPyTorch.utils.config.config_option import ConfigOption, to_bool
from autoPyTorch.training.base_training import BaseTrainingTechnique, BaseBatchLossComputationTechnique

from autoPyTorch.training.trainer import Trainer
from autoPyTorch.training.checkpoints.save_load import save_checkpoint, load_checkpoint, get_checkpoint_dir
from autoPyTorch.training.checkpoints.load_specific import load_model  # , load_optimizer, load_scheduler

torch.backends.cudnn.benchmark = True


class SimpleTrainNode(PipelineNode):
    def __init__(self):
        super(SimpleTrainNode, self).__init__()
        self.default_minimize_value = True
        self.logger = logging.getLogger('autonet')
        self.training_techniques = dict()
        self.batch_loss_computation_techniques = dict()
        self.add_batch_loss_computation_technique("standard", BaseBatchLossComputationTechnique)

    def fit(self, hyperparameter_config, pipeline_config,
            train_loader, valid_loader,
            network, optimizer, lr_scheduler,
            train_metric, additional_metrics,
            log_functions,
            budget,
            loss_function,
            budget_type,
            config_id, working_directory,
            train_indices, valid_indices):

        if budget < 1e-5:
            return {'loss': float('inf') if pipeline_config["minimize"] else -float('inf'), 'info': dict()}

        # prepare
        if not torch.cuda.is_available():
            pipeline_config["cuda"] = False

        device = pipeline_config['device']

        checkpoint_path = get_checkpoint_dir(working_directory)
        checkpoint = None
        if pipeline_config['save_checkpoints']:
            checkpoint = load_checkpoint(checkpoint_path, config_id, budget)

        network = load_model(network, checkpoint)

        tensorboard_logging = 'use_tensorboard_logger' in pipeline_config and pipeline_config['use_tensorboard_logger']

        hyperparameter_config = ConfigWrapper(self.get_name(), hyperparameter_config)

        batch_loss_name = hyperparameter_config[
            "batch_loss_computation_technique"] if "batch_loss_computation_technique" in hyperparameter_config else \
        pipeline_config["batch_loss_computation_techniques"][0]

        batch_loss_computation_technique = self.batch_loss_computation_techniques[batch_loss_name]()
        batch_loss_computation_technique.set_up(
            pipeline_config, ConfigWrapper(batch_loss_name, hyperparameter_config), self.logger)

        # Training loop
        logs = []
        epoch = 0

        train_metrics = []
        val_metrics = [train_metric] + additional_metrics
        if pipeline_config['evaluate_on_train_data']:
            train_metrics = val_metrics
        elif valid_loader is None:
            self.logger.warning(
                'No valid data specified and train process should not evaluate on train data! Will ignore \"evaluate_on_train_data\" and evaluate on train data!')
            train_metrics = val_metrics

        trainer = Trainer(
            model=network,
            loss_computation=batch_loss_computation_technique,
            criterion=loss_function,
            budget=budget,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            budget_type=budget_type,
            device=device,
            config_id=config_id,
            checkpoint_path=checkpoint_path if pipeline_config['save_checkpoints'] else None,
            images_to_plot=tensorboard_logging * pipeline_config['tensorboard_images_count'])

        model_params = self.count_parameters(network)

        network.param_list = []

        for name, param in network.named_parameters():
            network.param_list.append(copy.deepcopy(param))

        proxies = dict()

        last_log_time = time.time()
        while epoch < budget:
            # prepare epoch
            log = dict()

            if epoch == 0 and valid_loader is not None:
                valid_metric_results, proxies = trainer.evaluate(valid_loader, val_metrics, proxies, epoch=epoch)

                for i, metric in enumerate(val_metrics):
                    log['val_' + metric.__name__] = valid_metric_results[i]

            train_metric_results, train_loss, stop_training, proxies = trainer.train(epoch, train_loader, train_metrics,
                                                                                     proxies,
                                                                                     pipeline_config['proxy_names'])
            log['train_loss'] = train_loss
            for i, metric in enumerate(train_metrics):
                log['train_' + metric.__name__] = train_metric_results[i]

            for func in log_functions:
                log[func.__name__] = func(network, epoch + 1)

            print("TRAINER: log at epoch", epoch, ":", log)

            log['epochs'] = epoch + 1
            log['model_parameters'] = model_params
            log['learning_rate'] = optimizer.param_groups[0]['lr']

            logs.append(log)

            epoch += 1

            self.logger.debug("Epoch: " + str(epoch) + " : " + str(log))

            if 'params' not in proxies.keys():
                proxies['params'] = [log['model_parameters']]
            proxies['params'].append(log['model_parameters'])

            proxies['loss'].append(round(log['train_loss'], 3))
            proxies['train_acc'].append(round(log['train_accuracy'], 3))
            if valid_loader is not None:
                valid_metric_results, proxies = trainer.evaluate(valid_loader, val_metrics, proxies, epoch=epoch)

                for i, metric in enumerate(val_metrics):
                    log['val_' + metric.__name__] = valid_metric_results[i]
            proxies['train_cross_entropy'].append(round(log['train_cross_entropy'], 3))

            if tensorboard_logging and time.time() - last_log_time >= pipeline_config['tensorboard_min_log_interval']:
                import tensorboard_logger as tl
                worker_path = 'Train/'
                tl.log_value(worker_path + 'budget', float(budget), epoch)
                for name, value in log.items():
                    tl.log_value(worker_path + name, float(value), epoch)

        self.logger.debug("Finished Training")

        opt_metric_name = 'train_' + train_metric.__name__
        if valid_loader is not None:
            opt_metric_name = 'val_' + train_metric.__name__

        if pipeline_config["minimize"]:
            final_log = min(logs, key=lambda x: x[opt_metric_name])
        else:
            final_log = max(logs, key=lambda x: x[opt_metric_name])

        if tensorboard_logging:
            import tensorboard_logger as tl
            worker_path = 'Train/'
            tl.log_value(worker_path + 'budget', float(budget), epoch)
            for name, value in final_log.items():
                tl.log_value(worker_path + name, float(value), epoch)

        if trainer.latest_checkpoint:
            final_log['checkpoint'] = trainer.latest_checkpoint
        elif pipeline_config['save_checkpoints']:
            path = save_checkpoint(checkpoint_path, config_id, budget, network, optimizer, lr_scheduler)
            final_log['checkpoint'] = path

        final_log['train_datapoints'] = len(train_indices)
        if valid_loader is not None:
            final_log['val_datapoints'] = len(valid_indices)

        loss = final_log[opt_metric_name] * (1 if pipeline_config["minimize"] else -1)

        proxies['final_val_acc'] = final_log['val_accuracy']
        proxies['final_train_acc'] = final_log['train_accuracy']

        return {'loss': loss, 'info': final_log, 'proxies': proxies}

    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def predict(self, pipeline_config, network, predict_loader, dataset_info, train_metric):
        device = pipeline_config['device']

        if dataset_info.default_dataset:
            metric_results = Trainer(None, network, None, None, None, None, None, device).evaluate(predict_loader,
                                                                                                   [train_metric])
            return {'score': metric_results[0]}
        else:
            Y = predict(network, predict_loader, None, device)
            return {'Y': Y.detach().cpu().numpy()}

    def add_training_technique(self, name, training_technique):
        if (not issubclass(training_technique, BaseTrainingTechnique)):
            raise ValueError("training_technique type has to inherit from BaseTrainingTechnique")
        self.training_techniques[name] = training_technique

    def remove_training_technique(self, name):
        del self.training_techniques[name]

    def add_batch_loss_computation_technique(self, name, batch_loss_computation_technique):
        if (not issubclass(batch_loss_computation_technique, BaseBatchLossComputationTechnique)):
            raise ValueError(
                "batch_loss_computation_technique type has to inherit from BaseBatchLossComputationTechnique, got " + str(
                    batch_loss_computation_technique))
        self.batch_loss_computation_techniques[name] = batch_loss_computation_technique

    def remove_batch_loss_computation_technique(self, name, batch_loss_computation_technique):
        del self.batch_loss_computation_techniques[name]

    def get_hyperparameter_search_space(self, **pipeline_config):
        pipeline_config = self.pipeline.get_pipeline_config(**pipeline_config)
        cs = ConfigSpace.ConfigurationSpace()

        hp_batch_loss_computation = cs.add_hyperparameter(
            CSH.CategoricalHyperparameter("batch_loss_computation_technique",
                                          list(self.batch_loss_computation_techniques.keys())))

        for name, technique in self.batch_loss_computation_techniques.items():
            parent = {'parent': hp_batch_loss_computation,
                      'value': name} if hp_batch_loss_computation is not None else None
            cs.add_configuration_space(prefix=name,
                                       configuration_space=technique.get_hyperparameter_search_space(**pipeline_config),
                                       delimiter=ConfigWrapper.delimiter, parent_hyperparameter=parent)

        possible_loss_comps = sorted(set(pipeline_config["batch_loss_computation_techniques"]).intersection(
            self.batch_loss_computation_techniques.keys()))
        self._update_hyperparameter_range('batch_loss_computation_technique', possible_loss_comps, check_validity=False,
                                          override_if_already_modified=False)

        return self._apply_user_updates(cs)

    #    def resnet_criterion(self, config):
    #        nr_main_blocks = config["NetworkSelectorDatasetInfo:resnet:nr_main_blocks"]
    #        nr_res_blocks = ["NetworkSelectorDatasetInfo:resnet:nr_residual_blocks_" + str(mb) for mb in range(1, nr_main_blocks+1)]
    #        nr_res_branches = ["NetworkSelectorDatasetInfo:resnet:res_branches_" + str(mb) for mb in range(1, nr_main_blocks+1)]
    #
    #        limit = 8
    #        total_blocks = 0
    #
    #        for n_subblocks, n_res_branches in zip(nr_res_blocks, nr_res_branches):
    #            total_blocks += n_subblocks * n_res_branches
    #
    #        criterion = total_blocks>limit
    #        if criterion:
    #            print("ResNet too large", total_blocks, ">", limit)
    #
    #        return criterion

    def get_pipeline_config_options(self):
        options = [
            ConfigOption(name="batch_loss_computation_techniques",
                         default=list(self.batch_loss_computation_techniques.keys()),
                         type=str, list=True, choices=list(self.batch_loss_computation_techniques.keys())),
            ConfigOption("minimize", default=self.default_minimize_value, type=to_bool, choices=[True, False]),
            ConfigOption("cuda", default=True, type=to_bool, choices=[True, False]),
            ConfigOption("save_checkpoints", default=False, type=to_bool, choices=[True, False]),
            ConfigOption("tensorboard_min_log_interval", default=30, type=int),
            ConfigOption("tensorboard_images_count", default=0, type=int),
            ConfigOption("evaluate_on_train_data", default=True, type=to_bool),
        ]
        for name, technique in self.training_techniques.items():
            options += technique.get_pipeline_config_options()
        for name, technique in self.batch_loss_computation_techniques.items():
            options += technique.get_pipeline_config_options()
        return options


def predict(network, test_loader, metrics, device, move_network=True):
    """ predict batchwise """

    # Batch prediction
    network.eval()
    if metrics is not None:
        metric_results = [0] * len(metrics)
    N = 0.0
    for i, (X_batch, Y_batch) in tqdm(enumerate(test_loader), total=len(test_loader), desc="Evaluating Batches"):
        # Predict on batch
        X_batch = Variable(X_batch).to(device)
        batch_size = X_batch.size(0)

        out = network(X_batch)
        # in case of auxiliary head
        if isinstance(out, tuple):
            Y_batch_pred = out[0].detach().cpu()
        else:
            Y_batch_pred = out.detach().cpu()

        if metrics is None:
            # Infer prediction shape
            if i == 0:
                Y_pred = Y_batch_pred
            else:
                # Add to prediction tensor
                Y_pred = torch.cat((Y_pred, Y_batch_pred), 0)
        else:
            for i, metric in enumerate(metrics):
                metric_results[i] += metric(Y_batch, Y_batch_pred) * batch_size

        N += batch_size

    if metrics is None:
        return Y_pred
    else:
        return [res / N for res in metric_results]
