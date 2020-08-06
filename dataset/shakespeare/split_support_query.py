# 这个文件用于将客户端的各个数据拆分成 support 和 query.
import pickle
import numpy as np
import os
import sys
SEED = 0

ROOT = os.path.dirname(os.path.realpath(__file__))
PAR = os.path.dirname(os.path.dirname(ROOT))
PROJECT_DIR = os.path.dirname(os.path.dirname(ROOT))

sys.path.append(PAR)

from dataset.data_reader import read_data

train_prefix = f'{ROOT}/data/train'
test_prefix = f'{ROOT}/data/test'

if not os.path.exists(train_prefix):
    os.makedirs(train_prefix)
if not os.path.exists(test_prefix):
    os.makedirs(test_prefix)


def support_query_split(X, y, p):
    assert len(X) == len(y)
    np.random.seed(SEED)
    shuffled_indexes = np.random.permutation(len(X))
    sp_sz = int(len(X) * p)
    train_index = shuffled_indexes[:sp_sz]
    test_index = shuffled_indexes[sp_sz:]
    return X[train_index].tolist(), X[test_index].tolist(), y[train_index].tolist(), y[test_index].tolist()

def print_stats(data):
    """
    统计信息, 必须满足需要序列化的数据的格式
    :param data:
    :param max_class_num:
    :return:
    """
    y_size = []
    all_classes = set()
    for i, user in enumerate(data['users']):
        x, y = data['user_data'][user]['x'], data['user_data'][user]['y']
        num_data = data['num_samples'][i]
        assert num_data == len(y)
        y_unique = set(y)
        # y 个类别的数量
        y_size.append(len(y_unique))
        all_classes.update(y_unique)
    print('Client num:', len(data['users']), ', Samples:', sum(data['num_samples']), ', Classes:', len(all_classes))
    print('\tSample per client, std:', np.std(data['num_samples']), ', mean:', np.mean(data['num_samples']), ', min:', np.min(data['num_samples']), ', max:', np.max(data['num_samples']))
    print('\tClasses per client, min:', np.min(y_size), ', max:', np.max(y_size))

def split_to_support_query(train_dir, test_dir, p):
    """
    原来的数据格式为 :
    {
        <user_name>: {
            x: [],
            y: []
        }
    }
    转换后的结构也是这样, 但是保存的 train 则代表了 support, test 则为 query
    :param train_dir:
    :param test_dir:
    :param p:
    :return:
    """
    # Create data structure
    f_train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    f_test_data = {'users': [], 'user_data': {}, 'num_samples': []}
    before_merged = {'users': [], 'user_data': {}, 'num_samples': []}
    train_clients, train_groups, train_data, test_data = read_data(train_dir, test_dir, data_format='json')
    # 合并
    for user_name in train_clients:
        print('User', user_name, end=' ')
        train_x = train_data[user_name]['x']
        train_y = train_data[user_name]['y']
        test_x = test_data[user_name]['x']
        test_y = test_data[user_name]['y']
        # 合并之后
        print('Found, train size:', len(train_y), ', test size:', len(test_y), end=' ')
        all_x = np.asarray(train_x + test_x)
        all_y = np.asarray(train_y + test_y)
        #
        spt_x, qry_x, spt_y, qry_y = support_query_split(all_x, all_y, p=p)
        f_train_data['users'].append(user_name)
        f_train_data['user_data'][user_name] = {'x': spt_x, 'y': spt_y}
        f_train_data['num_samples'].append(len(spt_y))

        f_test_data['users'].append(user_name)
        f_test_data['user_data'][user_name] = {'x': qry_x, 'y': qry_y}
        f_test_data['num_samples'].append(len(qry_y))
        print('Query size:', len(qry_y), 'Support size:', len(spt_y))

        # for stats
        before_merged['users'].append(user_name)
        before_merged['user_data'][user_name] = {'x': all_x, 'y': all_y}
        before_merged['num_samples'].append(len(all_y))

    with open(os.path.join(train_prefix, f'p_{p}.pkl'), 'wb') as outfile:
        pickle.dump(f_train_data, outfile)
    with open(os.path.join(test_prefix, f'p_{p}.pkl'), 'wb') as outfile:
        pickle.dump(f_test_data, outfile)

    # 统计相关的信息
    print('拆分之前:')
    print_stats(data=before_merged)
    print('训练:')
    print_stats(data=f_train_data)
    print('测试:')
    print_stats(data=f_test_data)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: {0} <p>'.format(sys.argv[0]))
        exit(0)
    p = float(sys.argv[1])
    d = os.path.join(PROJECT_DIR, 'leaf', 'data', 'shakespeare', 'data', 'train')
    t = os.path.join(PROJECT_DIR, 'leaf', 'data', 'shakespeare', 'data', 'test')
    split_to_support_query(d, t, p)