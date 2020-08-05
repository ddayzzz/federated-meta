import pickle
import os
import json
from collections import defaultdict


__all__ = ['read_data', 'get_distributed_data_cfgs']


def _read_data_pkl(train_data_dir, test_data_dir, sub_data=None):
    """
    解析数据
    :param train_data_dir: 训练数据目录, 自动读取 pkl
    :param test_data_dir: 测试数据目录, 自动读取 pkl
    :return: clients的编号(按照升序), groups, train_data, test_data (两者均为dict, 键是 client 的编号; 映射为 x_index 表示索引, 这个依赖于原始数据集)
    """

    clients = []
    groups = []
    train_data_index = {}
    test_data_index = {}
    print('>>> Read data from:')

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.pkl')]
    if sub_data is not None:
        taf = sub_data + '.pkl'
        assert taf in train_files
        train_files = [taf]

    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        print('    ', file_path)

        with open(file_path, 'rb') as inf:
            cdata = pickle.load(inf)
        # 所有的用户
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        # user_data 是一个字典
        train_data_index.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.pkl')]
    if sub_data is not None:
        taf = sub_data + '.pkl'
        assert taf in test_files
        test_files = [taf]

    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        print('    ', file_path)

        with open(file_path, 'rb') as inf:
            cdata = pickle.load(inf)
        test_data_index.update(cdata['user_data'])

    clients = list(sorted(train_data_index.keys()))
    return clients, groups, train_data_index, test_data_index


def _read_dir_leaf(data_dir):
    print('>>> Read data from:', data_dir)
    clients = []
    groups = []
    # 如果 dict 对象不存在时候, 不raise一个KeyError
    data = defaultdict(lambda: None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data


def get_distributed_data_cfgs(data_name, sub_name, client_id):
    root = os.path.dirname(os.path.realpath(__file__))
    cfgs = os.path.join(root, data_name, 'data', 'distributed', sub_name)
    # all_cfgs = os.listdir(cfgs)
    return os.path.join(cfgs, client_id + '.json')


def read_data(train_data_dir, test_data_dir, data_format, sub_data=None):
    if data_format == 'json':
        # 这里的数据集不区分对应的格式
        assert sub_data is None, 'LEAF 格式的数据保存为多个 JSON 文件, 不能指定 subdata 名'
        train_clients, train_groups, train_data = _read_dir_leaf(train_data_dir)
        test_clients, test_groups, test_data = _read_dir_leaf(test_data_dir)

        assert train_clients == test_clients
        assert train_groups == test_groups

        return train_clients, train_groups, train_data, test_data
    elif data_format == 'pkl':
        return _read_data_pkl(train_data_dir, test_data_dir, sub_data)
    else:
        raise ValueError('仅仅支持两种格式的数据: *.pkl, *.json')