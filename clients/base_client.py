import tqdm
import torch
from torch.utils.data import DataLoader
from torch import nn


class BaseClient(object):

    def __init__(self, id, train_dataset, test_dataset, options, optimizer, model: nn.Module, model_flops, model_bytes):
        """
        定义客户端类型
        :param id: 客户端的 id
        :param worker: 客户端和模型连接器 Worker
        :param batch_size: mini-batch 所用的 batch 大小
        :param criterion: 分类任务的评价器
        :param train_dataset: 训练集
        :param test_dataset: 测试集/验证集
        """
        self.id = id
        # 这个必须是客户端相关的
        self.optimizer = optimizer
        self.num_train_data = len(train_dataset)
        self.num_test_data = len(test_dataset)
        self.num_epochs = options['num_epochs']
        self.num_batch_size = options['batch_size']
        self.options = options
        self.device = options['device']
        self.quiet = options['quiet']
        self.train_inner_step = options['train_inner_step']
        self.test_inner_step = options['test_inner_step']
        self.same_mini_batch = options['same_mini_batch']
        self.model = model

        self.train_dataset_loader = self.create_data_loader(train_dataset)
        self.test_dataset_loader = self.create_data_loader(test_dataset)
        if self.is_train_mini_batch:
            self.train_dataset_loader_iterator = iter(self.train_dataset_loader)
        if self.is_test_mini_batch:
            self.test_dataset_loader_iterator = iter(self.test_dataset_loader)

        #
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        #
        self.criterion = nn.CrossEntropyLoss(reduction='mean').to(self.device)
        self.flops = model_flops
        self.model_bytes = model_bytes

    @property
    def is_train_mini_batch(self):
        return self.train_inner_step > 0

    @property
    def is_test_mini_batch(self):
        return self.test_inner_step > 0

    def gen_train_batchs(self):
        """
        产生基于 dataloader 的若干的 mini-batch
        引用: https://github.com/pytorch/pytorch/issues/1917#issuecomment-433698337
        :param dataloader:
        :param mini_batch_gen_size: 产生多少次?
        :return:
        """
        for i in range(self.train_inner_step):
            try:
                data, target = next(self.train_dataset_loader_iterator)
            except StopIteration:
                self.train_dataset_loader_iterator = iter(self.train_dataset_loader)
                data, target = next(self.train_dataset_loader_iterator)
            yield data, target

    def gen_test_batchs(self):
        """
        产生基于 dataloader 的若干的 mini-batch
        :param dataloader:
        :param mini_batch_gen_size: 产生多少次?
        :return:
        """
        for i in range(self.test_inner_step):
            try:
                data, target = next(self.test_dataset_loader_iterator)
            except StopIteration:
                self.test_dataset_loader_iterator = iter(self.test_dataset_loader)
                data, target = next(self.test_dataset_loader_iterator)
            yield data, target

    def gen_train_batchs_use_same_batch(self):
        """
        产生基于 dataloader 的若干的 mini-batch
        引用: https://github.com/pytorch/pytorch/issues/1917#issuecomment-433698337
        :param dataloader:
        :param mini_batch_gen_size: 产生多少次?
        :return:
        """
        try:
            data, target = next(self.train_dataset_loader_iterator)
        except StopIteration:
            self.train_dataset_loader_iterator = iter(self.train_dataset_loader)
            data, target = next(self.train_dataset_loader_iterator)
        for i in range(self.train_inner_step):
            yield data, target

    def gen_test_batchs_use_same_batch(self):
        """
        产生基于 dataloader 的若干的 mini-batch
        :param dataloader:
        :param mini_batch_gen_size: 产生多少次?
        :return:
        """
        try:
            data, target = next(self.test_dataset_loader_iterator)
        except StopIteration:
            self.test_dataset_loader_iterator = iter(self.test_dataset_loader)
            data, target = next(self.test_dataset_loader_iterator)
        for i in range(self.test_inner_step):
            yield data, target

    def create_data_loader(self, dataset):
        return DataLoader(dataset, batch_size=self.num_batch_size, shuffle=True)

    def get_parameters_list(self):
        with torch.no_grad():
            ps = [p.data.clone().detach() for p in self.model.parameters()]
        return ps

    def set_parameters_list(self, params_list):
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), params_list):
                # 设置参数的值
                p.data.copy_(d.data)

    def test(self, data_loader):
        self.model.eval()
        train_loss = train_acc = train_total = 0
        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(data_loader):
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                loss = self.criterion(pred, y)
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum().item()

                target_size = y.size(0)
                # TODO 一般的损失函数会进行平均(mean), 但是这里不需要, 一种做法是指定损失函数仅仅用 sum, 但是考虑到pytorch中的损失函数默认为mean,故这里进行了些修改
                single_batch_loss = loss.item() * target_size
                train_loss += single_batch_loss
                train_acc += correct
                train_total += target_size
        return_dict = {"loss": train_loss / train_total,
                       "acc": train_acc / train_total,
                       'sum_loss': train_loss,
                       'sum_corrects': train_acc,
                       'num_samples': train_total}
        return return_dict

    def solve_epochs(self, round_i, client_id, data_loader, optimizer, num_epochs, hide_output: bool = False):
        device = self.device
        criterion = self.criterion
        self.model.train()

        with tqdm.trange(num_epochs, disable=hide_output) as t:
            train_loss = train_acc = train_total = 0
            for epoch in t:
                t.set_description(f'Client: {client_id}, Round: {round_i}, Epoch :{epoch}')
                for batch_idx, (X, y) in enumerate(data_loader):
                    # from IPython import embed
                    # embed()
                    X, y = X.to(device), y.to(device)

                    optimizer.zero_grad()
                    pred = self.model(X)

                    # if torch.isnan(pred.max()):
                    #     from IPython import embed
                    #     embed()

                    loss = criterion(pred, y)
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm(self.model.parameters(), 60)
                    optimizer.step()

                    _, predicted = torch.max(pred, 1)
                    correct = predicted.eq(y).sum().item()

                    target_size = y.size(0)
                    # TODO 一般的损失函数会进行平均(mean), 但是这里不需要, 一种做法是指定损失函数仅仅用 sum, 但是考虑到pytorch中的损失函数默认为mean,故这里进行了些修改
                    single_batch_loss = loss.item() * target_size
                    train_loss += single_batch_loss
                    train_acc += correct
                    train_total += target_size
                    if (batch_idx % 10 == 0):
                        # 纯数值, 这里使用平均的损失
                        t.set_postfix(mean_loss=loss.item())

        # 这一步非常关键, 清除优化器指向的全局模型的梯度
        optimizer.zero_grad()

        comp = num_epochs * train_total * self.flops
        return_dict = {"loss": train_loss / train_total,
                       "acc": train_acc / train_total,
                       'sum_loss': train_loss,
                       'sum_corrects': train_acc,
                       'num_samples': train_total}
        #
        bytes_w = self.model_bytes
        bytes_r = self.model_bytes
        flop_stats = {'id': self.id, 'bytes_w': bytes_w, 'comp': comp, 'bytes_r': bytes_r}
        return return_dict, flop_stats, self.get_parameters_list()