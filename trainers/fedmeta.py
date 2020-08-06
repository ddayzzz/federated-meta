import torch
from clients.base_client import BaseClient
from trainers.fedbase import BaseFedarated, MiniDataset
import pandas as pd
import numpy as np
from utils.flops_counter import get_model_complexity_info
from learn2learn.algorithms.maml import MAML
from learn2learn.algorithms.meta_sgd import MetaSGD


class Adam:

    """
    全局 Adam, 用来基于从客户端收集的梯度, 来更新全局网络的参数
    """

    def __init__(self, lr=0.01, betas=(0.9, 0.999), eps=1e-08):
        """

        :param lr:
        :param betas:
        :param eps:
        """
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.m = dict()
        self.v = dict()
        self.n = 0
        self.creted_momtem_grad_index = set()

    def __call__(self, params, grads, i):
        # 创建对应的 id
        if i not in self.m:
            self.m[i] = torch.zeros_like(params)
        if i not in self.v:
            self.v[i] = torch.zeros_like(params)

        self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads
        self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * torch.square(grads)

        alpha = self.lr * np.sqrt(1 - np.power(self.beta2, self.n))
        alpha = alpha / (1 - np.power(self.beta1, self.n))

        params.sub_(alpha * self.m[i] / (torch.sqrt(self.v[i]) + self.eps))

    def increase_n(self):
        self.n += 1


class Client(BaseClient):

    def __init__(self, id, train_dataset, test_dataset, options, optimizer, model, model_flops, model_bytes):
        # 这里的 model 即为 MAML 封装的对象
        self.meta_inner_step = options['meta_inner_step']
        self.store_to_cpu = options['store_to_cpu']
        super(Client, self).__init__(id, train_dataset, test_dataset, options, optimizer, model, model_flops, model_bytes)
        if self.is_mini_batch:
            self.train_dataset_loader_iterator = self.generate_batch_generator(self.train_dataset_loader)
            self.test_dataset_loader_iterator = self.generate_batch_generator(self.test_dataset_loader)

    @property
    def is_mini_batch(self):
        return self.meta_inner_step > 0

    def set_parameters_list(self, params_list: list):
        """
        :param params_list:
        :return:
        """
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), params_list):
                # 设置参数的值
                p.data.copy_(d.data)

    def get_parameters_list(self) -> list:
        with torch.no_grad():
            if self.store_to_cpu:
                ps = [p.data.cpu() for p in self.model.parameters()]
            else:
                ps = [p.data.clone().detach() for p in self.model.parameters()]
        return ps

    def count_correct(self, preds, targets):
        _, predicted = torch.max(preds, 1)
        correct = predicted.eq(targets).sum().item()
        return correct

    def solve_meta_one_epoch(self):
        if self.is_mini_batch:
            support_data_loader = self.gen_train_batchs()
            query_data_loader = self.gen_test_batchs()
        else:
            query_data_loader = self.test_dataset_loader
            support_data_loader = self.train_dataset_loader

        self.model.train()
        # 克隆之后, learn 为中间节点, 本身不带有梯度
        learner = self.model.clone()
        # 记录相关的信息
        support_loss, support_correct, support_num_sample = [], [], []
        for batch_idx, (x, y) in enumerate(support_data_loader):
            x, y = x.to(self.device), y.to(self.device)
            num_sample = y.size(0)
            pred = learner(x)
            loss = self.criterion(pred, y)
            # 评估
            correct = self.count_correct(pred, y)
            # 写入相关的记录, 这份 loss 是平均的
            support_loss.append(loss.item())
            support_correct.append(correct)
            support_num_sample.append(num_sample)
            # 计算 loss 关于当前参数的导数, 并更新目前网络的参数(回传到 model)
            learner.adapt(loss)

        # 此使的参数基于 query
        query_loss, query_correct, query_num_sample = [], [], []
        loss_sum = 0.0
        for batch_idx, (x, y) in enumerate(query_data_loader):
            x, y = x.to(self.device), y.to(self.device)
            num_sample = y.size(0)
            pred = learner(x)
            loss = self.criterion(pred, y)
            # batch_sum_loss
            # 评估
            correct = self.count_correct(pred, y)
            # 写入相关的记录, 这份 loss 是平均的
            query_loss.append(loss.item())
            query_correct.append(correct)
            query_num_sample.append(num_sample)
            #
            loss_sum += loss * num_sample

        spt_sz = np.sum(support_num_sample)
        qry_sz = np.sum(query_num_sample)
        mean_loss = loss_sum / qry_sz
        # 这个优化器的唯一作用是清除网络多余的梯度信息
        # self.optimizer.zero_grad()
        mean_loss.backward()
        # 获取此使的梯度, 这个梯度为一个 tensor
        if self.store_to_cpu:
            grads = [p.grad.data.cpu() for p in self.model.parameters()]
        else:
            grads = [p.grad.data.clone().detach() for p in self.model.parameters()]
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.zero_()
        #

        comp = (spt_sz + qry_sz) * self.flops
        bytes_w = self.model_bytes
        bytes_r = self.model_bytes
        flop_stats = {'id': self.id, 'bytes_w': bytes_w, 'comp': comp, 'bytes_r': bytes_r}

        return {
            'support_loss_sum': np.dot(support_loss, support_num_sample),
            'query_loss_sum': np.dot(query_loss, query_num_sample),
            'support_correct': np.sum(support_correct),
            'query_correct': np.sum(query_correct),
            'support_num_samples': spt_sz,
            'query_num_samples': qry_sz,
        }, flop_stats, grads

    def solve_meta_one_epoch_save_gpu_memory(self):
        if self.is_mini_batch:
            support_data_loader = self.gen_train_batchs()
            query_data_loader = self.gen_test_batchs()
        else:
            query_data_loader = self.test_dataset_loader
            support_data_loader = self.train_dataset_loader

        self.model.train()
        # 克隆之后, learn 为中间节点, 本身不带有梯度
        learner = self.model.clone()
        with torch.backends.cudnn.flags(enabled=False):
            # 记录相关的信息
            support_loss, support_correct, support_num_sample = [], [], []
            for batch_idx, (x, y) in enumerate(support_data_loader):
                x, y = x.to(self.device), y.to(self.device)
                num_sample = y.size(0)
                pred = learner(x)
                loss = self.criterion(pred, y)
                # 评估
                correct = self.count_correct(pred, y)
                # 写入相关的记录, 这份 loss 是平均的
                support_loss.append(loss.item())
                support_correct.append(correct)
                support_num_sample.append(num_sample)
                # 计算 loss 关于当前参数的导数, 并更新目前网络的参数(回传到 model)
                learner.adapt(loss)

            # 此使的参数基于 query
            query_loss, query_correct, query_num_sample = [], [], []
            for batch_idx, (x, y) in enumerate(query_data_loader):
                x, y = x.to(self.device), y.to(self.device)
                num_sample = y.size(0)
                pred = learner(x)
                loss = self.criterion(pred, y)
                # batch_sum_loss
                # 评估
                correct = self.count_correct(pred, y)
                # 写入相关的记录, 这份 loss 是平均的
                query_loss.append(loss.item())
                query_correct.append(correct)
                query_num_sample.append(num_sample)
                #
                (loss * num_sample).backward()

        spt_sz = np.sum(support_num_sample)
        qry_sz = np.sum(query_num_sample)

        # 获取此使的梯度, 这个梯度为一个 tensor
        grads = [p.grad.data.cpu() / qry_sz for p in self.model.parameters()]
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.zero_()
        #

        comp = (spt_sz + qry_sz) * self.flops
        bytes_w = self.model_bytes
        bytes_r = self.model_bytes
        flop_stats = {'id': self.id, 'bytes_w': bytes_w, 'comp': comp, 'bytes_r': bytes_r}

        return {
            'support_loss_sum': np.dot(support_loss, support_num_sample),
            'query_loss_sum': np.dot(query_loss, query_num_sample),
            'support_correct': np.sum(support_correct),
            'query_correct': np.sum(query_correct),
            'support_num_samples': spt_sz,
            'query_num_samples': qry_sz,
        }, flop_stats, grads


    def test_meta_one_epoch(self, train_loader, test_loader):
        # 这里觉得必须是有 adapt 的过程
        self.model.eval()
        learner = self.model.clone()
        # 清空目前指向的参数的梯度信息
        # 记录相关的信息
        support_loss, support_correct, support_num_sample = 0.0, 0, 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(self.device), y.to(self.device)
            num_sample = y.size(0)
            pred = learner(x)
            loss = self.criterion(pred, y)
            # 评估
            correct = self.count_correct(pred, y)
            # 写入相关的记录, 这份 loss 是平均的
            support_loss += loss.item() * num_sample
            support_correct += correct
            support_num_sample += num_sample
            # 计算 loss 关于当前参数的导数, 并更新目前网络的参数
            learner.adapt(loss)

        # 此使的参数基于 query
        query_loss, query_correct, query_num_sample = 0.0, 0, 0

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(test_loader):
                x, y = x.to(self.device), y.to(self.device)
                num_sample = y.size(0)
                pred = learner(x)
                loss = self.criterion(pred, y)
                # batch_sum_loss
                # 评估
                correct = self.count_correct(pred, y)
                # 写入相关的记录, 这份 loss 是平均的
                query_loss += loss.item() * num_sample
                query_correct += correct
                query_num_sample += num_sample

        return support_loss, support_correct, support_num_sample, query_loss, query_correct, query_num_sample

    def generate_batch_generator(self, dataloader):
        # 返回 dataloader 的迭代器
        return iter(dataloader)

    def gen_train_batchs(self):
        """
        产生基于 dataloader 的若干的 mini-batch
        引用: https://github.com/pytorch/pytorch/issues/1917#issuecomment-433698337
        :param dataloader:
        :param mini_batch_gen_size: 产生多少次?
        :return:
        """
        for i in range(self.meta_inner_step):
            try:
                data, target = next(self.train_dataset_loader_iterator)
            except StopIteration:
                self.train_dataset_loader_iterator = self.generate_batch_generator(self.train_dataset_loader)
                data, target = next(self.train_dataset_loader_iterator)
            yield data, target

    def gen_test_batchs(self):
        """
        产生基于 dataloader 的若干的 mini-batch
        :param dataloader:
        :param mini_batch_gen_size: 产生多少次?
        :return:
        """
        for i in range(self.meta_inner_step):
            try:
                data, target = next(self.test_dataset_loader_iterator)
            except StopIteration:
                self.test_dataset_loader_iterator = self.generate_batch_generator(self.test_dataset_loader)
                data, target = next(self.test_dataset_loader_iterator)
            yield data, target


class FedMeta(BaseFedarated):

    def __init__(self, options, model, read_dataset, more_metric_to_train=None):
        self.meta_algo = options['meta_algo']
        self.outer_lr = options['outer_lr']
        self.meta_inner_step = options['meta_inner_step']
        self.meta_train_test_split = options['meta_train_test_split']
        self.store_to_cpu = options['store_to_cpu']
        self.outer_opt = Adam(lr=self.outer_lr)

        if self.meta_inner_step <= 0:
            print('>>> Using FedMeta')
        else:
            print('>>> Using FedMeta, meta-train inner step: ', self.meta_inner_step)
        if self.meta_inner_step > 0:
            a = f'outerlr[{self.outer_lr}]_metaalgo[{self.meta_algo}]_minibatch[{self.meta_inner_step}]'
        else:
            a = f'outerlr[{self.outer_lr}]_metaalgo[{self.meta_algo}]'
        self.maml = MAML(lr=options['lr'], model=model.to(options['device']))
        super(FedMeta, self).__init__(options=options, model=self.maml, read_dataset=read_dataset, append2metric=a,
                                      more_metric_to_train=['query_acc', 'query_loss'])
        # 拆分客户端
        self.split_train_validation_test_clients()
        # self.train_support, self.train_query = self.generate_batch_generator(self.train_clients)
        self.model = None

    def setup_model(self, options, model):
        dev = options['device']
        input_shape = model.module.input_shape
        input_type = model.module.input_type if hasattr(model.module, 'input_type') else None
        self.flops, self.params_num, self.model_bytes = \
            get_model_complexity_info(model.module, input_shape, input_type=input_type, device=dev)
        return model

    def split_train_validation_test_clients(self, train_rate=0.8, val_rate=0.1):
        if self.meta_train_test_split <= 0:
            np.random.seed(self.options['seed'])
            train_rate = int(train_rate * self.num_clients)
            val_rate = int(val_rate * self.num_clients)
            test_rate = self.num_clients - train_rate - val_rate

            assert test_rate > 0 and val_rate > 0 and test_rate > 0, '不能为空'

            ind = np.random.permutation(self.num_clients)
            arryed_cls = np.asarray(self.clients)
            self.train_clients = arryed_cls[ind[:train_rate]].tolist()
            self.eval_clients = arryed_cls[ind[train_rate:train_rate + val_rate]].tolist()
            self.test_clients = arryed_cls[ind[train_rate + val_rate:]].tolist()

            print('用于训练的客户端数量{}, 用于验证:{}, 用于测试: {}'.format(len(self.train_clients), len(self.eval_clients),
                                                           len(self.test_clients)))
            return
        #
        self.train_clients = self.clients[:self.meta_train_test_split]
        self.test_clients = self.clients[self.meta_train_test_split:]
        print('用于 Meta-train客户端的数量{}, 用于Meta-test客户端数量: {}'.format(len(self.train_clients), len(self.test_clients)))

    def setup_clients(self, dataset, model):
        users, groups, train_data, test_data = dataset
        if len(groups) == 0:
            groups = [None for _ in users]
        dataset_wrapper = self.choose_dataset_wapper()
        all_clients = []
        for user, group in zip(users, groups):
            # if isinstance(user, str) and len(user) >= 5:
            #     user_id = int(user[-5:])
            # else:
            #     user_id = int(user)
            tr = dataset_wrapper(train_data[user], options=self.options)
            te = dataset_wrapper(test_data[user], options=self.options)
            c = Client(id=user, options=self.options, train_dataset=tr, test_dataset=te, optimizer=None, model=self.maml, model_flops=self.flops, model_bytes=self.model_bytes)
            all_clients.append(c)
        return all_clients

    def select_clients(self, round_i, num_clients=20):
        num_clients = min(num_clients, len(self.clients))
        np.random.seed(round_i)  # 确定每一轮次选择相同的客户端(用于比较不同算法在同一数据集下的每一轮的客户端不变)
        return np.random.choice(self.train_clients, num_clients, replace=False).tolist()

    def eval_on(self, round_i, clients, use_test_data=True, use_train_data=True, use_val_data=False):
        # 设置写入的数据
        df = pd.DataFrame(columns=['client_id', 'spt_acc', 'spt_loss', 'spt_size', 'qry_size', 'qry_acc', 'qry_loss'])

        spt_corr, spt_loss, spt_sz = 0, 0.0, 0
        qry_corr, qry_loss, qry_sz = 0, 0.0, 0
        for c in clients:
            c.set_parameters_list(self.latest_model)
            support_loss, support_correct, support_num_sample, query_loss, query_correct, query_num_sample = c.test_meta_one_epoch(c.train_dataset_loader, c.test_dataset_loader)
            spt_corr += support_correct
            spt_loss += support_loss
            spt_sz += support_num_sample
            qry_sz += query_num_sample
            qry_corr += query_correct
            qry_loss += query_loss
            #
            df = df.append({'client_id': c.id,
                            'spt_acc': support_correct / support_num_sample,
                            'spt_loss': support_loss / support_num_sample,
                            'qry_acc': query_correct / query_num_sample,
                            'qry_loss': query_loss / query_num_sample,
                            'qry_size': query_num_sample,
                            'spt_size': support_num_sample,}, ignore_index=True)

        fn = 'eval_at_round_{}.csv'.format(round_i)
        mean_spt_loss, mean_qry_loss = spt_loss / spt_sz, qry_loss / qry_sz
        mean_spt_acc, mean_qry_acc = spt_corr / spt_sz, qry_corr / qry_sz
        if not self.quiet:
            print(f'Round {round_i}, eval on meta-test client mean spt loss: {mean_spt_loss:.5f}, mean spt acc: {mean_spt_acc:.3%}', end='; ')
            print(f'mean qry loss: {mean_qry_loss:.5f}, mean qry acc: {mean_qry_acc:.3%}')
        self.metrics.update_eval_stats(round_i, df=df, on_which='meta-test', filename=fn, other_to_logger={
            'spt_loss': mean_spt_loss, 'spt_acc': mean_spt_acc, 'qry_acc': mean_qry_acc, 'qry_loss': mean_qry_loss
        })

    def solve_epochs(self, round_i, clients, epoch=None):
        spt_corrects = 0
        spt_loss = 0.0
        qry_loss = 0.0
        qry_corrects = 0
        qry_sz, spt_sz = 0, 0

        solns = []
        num_qry_size = []
        for c in clients:
            c.set_parameters_list(self.latest_model)
            # 保存信息
            if self.store_to_cpu:
                stat, flop_stat, grads = c.solve_meta_one_epoch_save_gpu_memory()
            else:
                stat, flop_stat, grads = c.solve_meta_one_epoch()
            # 总共正确的个数
            spt_corrects += stat['support_correct']
            qry_corrects += stat['query_correct']
            # loss 和
            spt_loss += stat['support_loss_sum']
            qry_loss += stat['query_loss_sum']
            #
            spt_sz += stat['support_num_samples']
            qry_sz += stat['query_num_samples']
            num_qry_size.append(stat['query_num_samples'])
            solns.append(grads)
            # 写入测试的相关信息
            self.metrics.update_commu_stats(round_i, flop_stat)

        mean_spt_loss, mean_qry_loss = spt_loss / spt_sz, qry_loss / qry_sz
        mean_spt_acc, mean_qry_acc = spt_corrects / spt_sz, qry_corrects / qry_sz

        stats = {
            'acc': mean_spt_acc, 'loss': mean_spt_loss,
            'query_acc': mean_qry_acc, 'query_loss': mean_qry_loss
        }
        if not self.quiet:
            print(f'Round {round_i}, meta-train, mean spt loss: {mean_spt_loss:.5f}, mean spt acc: {mean_spt_acc:.3%}', end='; ')
            print(f'mean qry loss: {mean_qry_loss:.5f}, mean qry acc: {mean_qry_acc:.3%}')

        self.metrics.update_train_stats_only_acc_loss(round_i, stats)
        return solns, num_qry_size

    def aggregate_grads_simple(self, solns, lr, weights_before):
        # 使用 adam
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
        # 普通的梯度下降 [u - (v * self.outer_lr / m) for u, v in zip(weights_before, g)]
        self.outer_opt.increase_n()
        for i in range(len(weights_before)):
            # 这是一个 in-place 的函数
            self.outer_opt(weights_before[i], g[i] / m, i=i)

    def aggregate_grads_weighted(self, solns, num_samples, weights_before):
        # 使用 adam
        m = len(solns)
        g = []
        for i in range(len(solns[0])):
            # i 表示的当前的梯度的 index
            # 总是 client 1 的梯度的形状
            grad_sum = torch.zeros_like(solns[0][i])
            total_sz = 0
            for ic, sz in enumerate(num_samples):
                grad_sum += solns[ic][i] * sz
                total_sz += sz
                # 累加之后, 进行梯度下降
            g.append(grad_sum / total_sz)
        # 普通的梯度下降 [u - (v * self.outer_lr / m) for u, v in zip(weights_before, g)]
        self.outer_opt.increase_n()
        for i in range(len(weights_before)):
            # 这是一个 in-place 的函数
            self.outer_opt(weights_before[i], g[i], i=i)

    def aggregate(self, solns, weight_before, num_qry_samples):
        # self.aggregate_grads_simple(solns=solns, weights_before=weight_before, lr=None)
        self.aggregate_grads_weighted(solns=solns, weights_before=weight_before, num_samples=num_qry_samples)

    def train(self):
        for round_i in range(self.num_rounds):
            print(f'>>> Global Training Round : {round_i}')

            selected_client_indices = self.select_clients(round_i=round_i, num_clients=self.clients_per_round)
            weight_before = self.latest_model
            solns, qry_num = self.solve_epochs(round_i=round_i, clients=selected_client_indices)

            self.aggregate(solns, weight_before=weight_before, num_qry_samples=qry_num)
            # eval on test
            if (round_i + 1) % self.eval_on_test_every_round == 0:
                self.eval_on(round_i=round_i, clients=self.test_clients)

            if (round_i + 1) % self.save_every_round == 0:
                # self.save_model(round_i)
                self.metrics.write()
        self.metrics.write()
