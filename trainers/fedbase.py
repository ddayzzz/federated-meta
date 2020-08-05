import numpy as np
import os
import time
import abc
import torch
from torch import nn, optim
import pandas as pd
from clients.base_client import BaseClient
from utils.metrics import Metrics
from utils.data_utils import MiniDataset
from utils.flops_counter import get_model_complexity_info



class BaseFedarated(abc.ABC):

    def __init__(self, options, model: nn.Module, read_dataset, append2metric=None, more_metric_to_train=None):
        """
        定义联邦学习的基本的服务器, 这里的模型是在所有的客户端之间共享使用
        :param options: 参数配置
        :param model: 模型
        :param dataset: 数据集参数
        :param optimizer: 优化器
        :param criterion: 损失函数类型(交叉熵,Dice系数等等)
        :param worker: Worker 实例
        :param append2metric: 自定义metric
        """
        self.model = self.setup_model(options=options, model=model)
        self.device = options['device']
        # 记录总共的训练数据
        self.options = options
        self.clients = self.setup_clients(dataset=read_dataset, model=model)
        self.num_epochs = options['num_epochs']
        self.num_rounds = options['num_rounds']
        self.clients_per_round = options['clients_per_round']
        self.save_every_round = options['save_every']
        self.eval_on_train_every_round = options['eval_on_train_every']
        self.eval_on_test_every_round = options['eval_on_test_every']
        self.eval_on_validation_every_round = options['eval_on_validation_every']
        self.num_clients = len(self.clients)
        # 使用 client 的API
        self.latest_model = self.clients[0].get_parameters_list()
        self.name = '_'.join(['', f'wn{options["clients_per_round"]}', f'tn{self.num_clients}'])
        self.metrics = Metrics(clients=self.clients, options=options, name=self.name, append2suffix=append2metric, result_prefix=options['result_prefix'], train_metric_extend_columns=more_metric_to_train)
        self.quiet = options['quiet']

    def setup_model(self, options, model):
        dev = options['device']
        model = model.to(dev)
        input_shape = model.input_shape
        input_type = model.input_type if hasattr(model, 'input_type') else None
        self.flops, self.params_num, self.model_bytes = \
            get_model_complexity_info(model, input_shape, input_type=input_type, device=dev)
        return model

    def setup_clients(self, dataset, model):
        users, groups, train_data, test_data = dataset
        if len(groups) == 0:
            groups = [None for _ in users]
        dataset_wrapper = MiniDataset
        all_clients = []
        for user, group in zip(users, groups):
            # if isinstance(user, str) and len(user) >= 5:
            #     user_id = int(user[-5:])
            # else:
            #     user_id = int(user)
            tr = dataset_wrapper(train_data[user], options=self.options)
            te = dataset_wrapper(test_data[user], options=self.options)
            opt = optim.SGD(self.model.parameters(), lr=self.options['lr'], momentum=0.5)
            c = BaseClient(id=user, options=self.options, train_dataset=tr, test_dataset=te, optimizer=opt, model=model, model_flops=self.flops, model_bytes=self.model_bytes)
            all_clients.append(c)
        return all_clients

    @abc.abstractmethod
    def train(self, *args, **kwargs):
        pass

    def select_clients(self, round_i, num_clients):
        """
        选择客户端, 采用的均匀的无放回采样
        :param round:
        :param num_clients:
        :return:
        """
        num_clients = min(num_clients, self.num_clients)
        np.random.seed(round_i)  # 确定每一轮次选择相同的客户端(用于比较不同算法在同一数据集下的每一轮的客户端不变)
        return np.random.choice(self.clients, num_clients, replace=False).tolist()

    def aggregate_parameters_weighted(self, solns, num_samples):
        """
        聚合模型
        :param solns: 列表.
        :param kwargs:
        :return: 聚合后的参数
        """
        # averaged_solution = torch.zeros_like(self.latest_model)
        # num_all_samples = 0
        # for num_sample, local_solution in solns:
        #     num_all_samples += num_sample
        #     averaged_solution += local_solution * num_sample # 加和, 乘以对应客户端的样本数量
        # averaged_solution /= num_all_samples  # 除以运行样本的整体数量
        # return averaged_solution.detach()
        lastes = []
        params_num = len(solns[0])
        m = len(solns)
        for p in range(params_num):
            new = torch.zeros_like(solns[0][p].data)
            sz = 0
            for num_sample, sol in zip(num_samples, solns):
                new += sol[p].data * num_sample
                sz += num_sample
            new /= sz
            lastes.append(new)
        return lastes

    def aggregate_grads_weights(self, solns, lr, num_samples, weights_before):
        """
        合并梯度, 这个和 fedavg 不相同
        :param grads:
        :return:
        """
        m = len(solns)
        g = []
        for i in range(len(solns[0])):
            # i 表示的当前的梯度的 index
            # 总是 client 1 的梯度的形状
            grad_sum = torch.zeros_like(solns[0][i])
            all_sz = 0
            for ic, sz in enumerate(num_samples):
                grad_sum += solns[ic][i] * sz
                all_sz += sz
            # 累加之后, 进行梯度下降
            g.append(grad_sum / all_sz)
        return [u - (v * lr) for u, v in zip(weights_before, g)]

    def aggregate_grads_simple(self, solns, lr, weights_before):
        """
        合并梯度(直接合并后除以参数的数量), 这个和 fedavg 不相同
        :param grads:
        :return:
        """
        m = len(solns)
        g = []
        for i in range(len(solns[0])):
            # i 表示的当前的梯度的 index
            # 总是 client 1 的梯度的形状
            grad_sum = torch.zeros_like(solns[0][i])
            for ic in range(m):
                grad_sum += solns[ic][i]
            # 累加之后, 进行梯度下降
            g.append(grad_sum)
        new_weights = [u - (v * lr / m) for u, v in zip(weights_before, g)]
        return new_weights

    @abc.abstractmethod
    def aggregate(self, *args, **kwargs):
        pass

    def eval_on(self, round_i, clients, use_test_data=False, use_train_data=False, use_val_data=False):
        assert use_test_data + use_train_data + use_val_data == 1, "不能同时设置"
        df = pd.DataFrame(columns=['client_id', 'mean_acc', 'mean_loss', 'num_samples'])

        num_samples = []
        tot_corrects = []
        losses = []
        for c in clients:
            # 设置网络
            c.set_parameters_list(self.latest_model)
            if use_test_data:
                stats =c.test(c.test_dataset_loader)
            elif use_train_data:
                stats = c.test(c.train_dataset_loader)
            elif use_val_data:
                stats = c.test(c.validation_dataset_loader)

            tot_corrects.append(stats['sum_corrects'])
            num_samples.append(stats['num_samples'])
            losses.append(stats['sum_loss'])
            #
            df = df.append({'client_id': c.id, 'mean_loss': stats['loss'], 'mean_acc': stats['acc'], 'num_samples': stats['num_samples'], }, ignore_index=True)

        # ids = [c.id for c in self.clients]
        # groups = [c.group for c in self.clients]
        mean_loss = sum(losses) / sum(num_samples)
        mean_acc = sum(tot_corrects) / sum(num_samples)
        #
        if use_test_data:
            fn, on = 'eval_on_test_at_round_{}.csv'.format(round_i), 'test'
        elif use_train_data:
            fn, on = 'eval_on_train_at_round_{}.csv'.format(round_i), 'train'
        elif use_val_data:
            fn, on = 'eval_on_validation_at_round_{}.csv'.format(round_i), 'validation'
        #
        if not self.quiet:
            print(f'Round {round_i}, eval on "{on}" dataset mean loss: {mean_loss:.5f}, mean acc: {mean_acc:.3%}')
        self.metrics.update_eval_stats(round_i, df, filename=fn, on_which=on, other_to_logger={'acc': mean_acc, 'loss': mean_loss})

    def solve_epochs(self, round_i, clients, num_epochs=None):
        if num_epochs is None:
            num_epochs = self.num_epochs

        num_samples = []
        tot_corrects = []
        losses = []

        solns = []
        for c in clients:
            c.set_parameters_list(self.latest_model)
            # 保存信息
            stat, flop_stat, soln = c.solve_epochs(round_i, c.id, c.train_dataset_loader, c.optimizer, num_epochs, hide_output=self.quiet)
            tot_corrects.append(stat['sum_corrects'])
            num_samples.append(stat['num_samples'])
            losses.append(stat['sum_loss'])
            #
            solns.append(soln)
            # 写入测试的相关信息
            self.metrics.update_commu_stats(round_i, flop_stat)

        mean_loss = sum(losses) / sum(num_samples)
        mean_acc = sum(tot_corrects) / sum(num_samples)

        stats = {
            'acc': mean_acc, 'loss': mean_loss,
        }
        if not self.quiet:
            print(f'Round {round_i}, train metric mean loss: {mean_loss:.5f}, mean acc: {mean_acc:.3%}')
        self.metrics.update_train_stats_only_acc_loss(round_i, stats)
        return solns, num_samples

    # 废弃
    def test_latest_model_on_traindata(self, round_i):
        """
        在训练数据集上测试
        :param round_i:
        :return:
        """
        begin_time = time.time()
        stats_from_train_data = self.local_test(use_eval_data=False)

        # 记录梯度用
        # flatten后的模型长度
        model_len = len(self.latest_model)
        global_grads = np.zeros(model_len)
        num_samples = []
        local_grads = []

        for c in self.clients:
            (num, client_grad), stat = c.solve_grad()
            local_grads.append(client_grad)
            num_samples.append(num)
            global_grads += client_grad * num
        global_grads /= np.sum(np.asarray(num_samples))
        stats_from_train_data['gradnorm'] = np.linalg.norm(global_grads)

        # 计算公式为 (客户端模型 - 上一次聚合后的模型) ^ 2, 一定程度上, 上一次聚合后的模型为平均的模型, 解释为方差
        difference = 0.
        for idx in range(len(self.clients)):
            difference += np.sum(np.square(global_grads - local_grads[idx]))
        difference /= len(self.clients)
        stats_from_train_data['graddiff'] = difference
        end_time = time.time()

        self.metrics.update_train_stats(round_i, stats_from_train_data)
        if self.print_result:
            print('\n>>> Round: {: >4d} / Acc: {:.3%} / Loss: {:.4f} /'
                  ' Grad Norm: {:.4f} / Grad Diff: {:.4f} / Time: {:.2f}s'.format(
                   round_i, stats_from_train_data['acc'], stats_from_train_data['loss'],
                   stats_from_train_data['gradnorm'], difference, end_time-begin_time))
        return global_grads



    def test_latest_model_on_evaldata(self, round_i):
        begin_time = time.time()
        stats_from_eval_data = self.local_test(use_eval_data=True)
        end_time = time.time()

        if self.print_result:
            print('>>> Test on eval: round: {} / acc: {:.3%} / '
                  'loss: {:.4f} / Time: {:.2f}s'.format(
                   round_i, stats_from_eval_data['acc'],
                   stats_from_eval_data['loss'], end_time-begin_time))
            # print('=' * 102 + "\n")

        self.metrics.update_eval_stats(round_i, stats_from_eval_data)

    def test_latest_model_on_traindata_only_acc_loss(self, round_i):
        begin_time = time.time()
        stats_from_eval_data = self.local_test(use_eval_data=False)
        end_time = time.time()

        if self.print_result:
            print('>>> Test on train: round: {} / acc: {:.3%} / '
                  'loss: {:.4f} / Time: {:.2f}s'.format(
                   round_i, stats_from_eval_data['acc'],
                   stats_from_eval_data['loss'], end_time-begin_time))
            # print('=' * 102 + "\n")

        self.metrics.update_train_stats_only_acc_loss(round_i, stats_from_eval_data)


    def save_model(self, round_i):
        self.worker.save(path=os.path.sep.join((self.metrics.result_path, self.metrics.exp_name, f'model_at_round_{round_i}.pkl')))
