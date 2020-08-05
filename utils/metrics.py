import os
import numpy as np
import pandas as pd
import json
from torch.utils.tensorboard import SummaryWriter
import time


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


class Metrics(object):

    def __init__(self, clients, options, name='', append2suffix=None, result_prefix='./result', train_metric_extend_columns=None):
        self.options = options
        num_rounds = options['num_rounds'] + 1

        self.bytes_written = {c.id: [0] * num_rounds for c in clients}
        self.client_computations = {c.id: [0] * num_rounds for c in clients}
        self.bytes_read = {c.id: [0] * num_rounds for c in clients}

        # # Statistics in training procedure
        # self.loss_on_train_data = [0] * num_rounds
        # self.acc_on_train_data = [0] * num_rounds
        # # self.gradnorm_on_train_data = [0] * num_rounds
        # # self.graddiff_on_train_data = [0] * num_rounds
        #
        # # Statistics in test procedure
        # self.loss_on_test_data = [0] * num_rounds
        # self.acc_on_test_data = [0] * num_rounds
        # 用来保存训练客户端时输出的信息, 这是 rounds 维度上的
        if train_metric_extend_columns is not None:
            assert isinstance(train_metric_extend_columns, list)
        else:
            train_metric_extend_columns = []
        self.train_metric_writer = pd.DataFrame(columns=['loss', 'acc'] + train_metric_extend_columns)
        # 记录训练的信息
        # customs
        self.customs_data = dict()
        self.num_rounds = num_rounds
        self.result_path = mkdir(os.path.join(result_prefix, self.options['dataset']))
        # suffix = '{}_sd{}_lr{}_ep{}_bs{}_{}'.format(name,
        #                                             options['seed'],
        #                                             options['lr'],
        #                                             options['num_epoch'],
        #                                             options['batch_size'],
        #                                             'w' if options['noaverage'] else 'a')
        suffix = '{}_sd{}_lr{}_ep{}_bs{}_wd{}'.format(name,
                                                      options['seed'],
                                                      options['lr'],
                                                      options['num_epochs'],
                                                      options['batch_size'],
                                                      options['wd'])
        if append2suffix is not None:
            suffix += '_' + append2suffix

        self.exp_name = '{}_{}_{}_{}'.format(time.strftime('%Y-%m-%dT%H-%M-%S'), options['algo'],
                                             options['model'], suffix)
        # if options['dis']:
        #     suffix = options['dis']
        #     self.exp_name += '_{}'.format(suffix)
        train_event_folder = mkdir(os.path.join(self.result_path, self.exp_name, 'train.event'))
        eval_event_folder = mkdir(os.path.join(self.result_path, self.exp_name, 'eval.event'))
        self.eval_metric_folder = mkdir(os.path.join(self.result_path, self.exp_name, 'eval_metric'))
        self.train_writer = SummaryWriter(train_event_folder)
        self.eval_writer = SummaryWriter(eval_event_folder)

    def update_commu_stats(self, round_i, stats):
        cid, bytes_w, comp, bytes_r = \
            stats['id'], stats['bytes_w'], stats['comp'], stats['bytes_r']

        self.bytes_written[cid][round_i] += bytes_w
        self.client_computations[cid][round_i] += comp
        self.bytes_read[cid][round_i] += bytes_r

    def extend_commu_stats(self, round_i, stats_list):
        for stats in stats_list:
            self.update_commu_stats(round_i, stats)

    # def update_train_stats(self, round_i, train_stats):
    #     self.loss_on_train_data[round_i] = train_stats['loss']
    #     self.acc_on_train_data[round_i] = train_stats['acc']
    #     self.gradnorm_on_train_data[round_i] = train_stats['gradnorm']
    #     self.graddiff_on_train_data[round_i] = train_stats['graddiff']
    #
    #     self.train_writer.add_scalar('train_loss', train_stats['loss'], round_i)
    #     self.train_writer.add_scalar('train_acc', train_stats['acc'], round_i)
    #     self.train_writer.add_scalar('gradnorm', train_stats['gradnorm'], round_i)
    #     self.train_writer.add_scalar('graddiff', train_stats['graddiff'], round_i)

    def update_grads_stats(self, round_i, stats):
        self.gradnorm_on_train_data[round_i] = stats['gradnorm']
        self.graddiff_on_train_data[round_i] = stats['graddiff']
        self.train_writer.add_scalar('gradnorm', stats['gradnorm'], round_i)
        self.train_writer.add_scalar('graddiff', stats['graddiff'], round_i)

    def update_train_stats_only_acc_loss(self, round_i, train_stats):
        self.train_metric_writer = self.train_metric_writer.append(train_stats, ignore_index=True)
        for k, v in train_stats.items():
            self.train_writer.add_scalar('train' + k, v, round_i)

    def update_eval_stats(self, round_i, df, on_which, filename, other_to_logger):
        df.to_csv(os.path.join(self.eval_metric_folder, filename))
        for k, v in other_to_logger.items():
            self.eval_writer.add_scalar(f'on_{on_which}_{k}', v, round_i)

    def update_custom_scalars(self, round_i, **data):
        for key, scalar in data.items():
            if key not in self.customs_data:
                self.customs_data[key] = [0] * self.num_rounds
            self.customs_data[key][round_i] = scalar
            self.train_writer.add_scalar(key, scalar_value=scalar, global_step=round_i)

    def write(self):
        metrics = dict()

        # Dict(key=cid, value=list(stats for each round))
        metrics['client_computations'] = self.client_computations
        metrics['bytes_written'] = self.bytes_written
        metrics['bytes_read'] = self.bytes_read
        for key, data in self.customs_data.items():
            metrics[key] = data
        metrics_dir = os.path.join(self.result_path, self.exp_name, 'metrics.json')
        params_dir = os.path.join(self.result_path, self.exp_name, 'params.json')
        with open(metrics_dir, 'w') as ouf:
            json.dump(metrics, ouf)

        with open(params_dir, 'w') as ouf:
            json.dump(self.options, ouf)

        self.train_metric_writer.to_csv(os.path.join(self.result_path, self.exp_name, 'train_metric.csv'))
