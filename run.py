import numpy as np
import importlib
import torch
import os
import random
from dataset.data_reader import read_data
from config import DATASETS, TRAINERS, MODEL_CONFIG
from config import base_options, add_dynamic_options


def read_options():
    parser = base_options()
    parser = add_dynamic_options(parser)
    parsed = parser.parse_args()
    options = parsed.__dict__
    # 设置种子
    os.environ['PYTHONHASHSEED'] = str(options['seed'])
    np.random.seed(1 + options['seed'])
    torch.manual_seed(12 + options['seed'])
    random.seed(1234 + options['seed'])
    if options['device'].startswith('cuda'):
        torch.cuda.manual_seed_all(123 + options['seed'])
        torch.backends.cudnn.deterministic = True  # cudnn


    # 读取数据集
    idx = options['dataset'].find("_")
    if idx != -1:
        dataset_name, sub_data = options['dataset'][:idx], options['dataset'][idx+1:]
    else:
        dataset_name, sub_data = options['dataset'], None
    assert dataset_name in DATASETS, "{} not in dataset {}!".format(dataset_name, DATASETS)

    # 将配置的参数添加到测试文件中
    model_cfg_key ='.'.join((dataset_name, options['model']))
    model_cfg = MODEL_CONFIG.get(model_cfg_key)

    # 加载选择的 solver 类
    trainer_path = 'trainers.%s' % options['algo']
    mod = importlib.import_module(trainer_path)
    trainer_class = getattr(mod, TRAINERS[options['algo']])

    # 加载模型类
    model_path = 'models.{0}.{1}'.format(dataset_name, options['model'])
    mod = importlib.import_module(model_path)
    model_obj = getattr(mod, 'Model')(**model_cfg)

    # 打印参数
    max_length = max([len(key) for key in options.keys()])
    fmt_string = '\t%' + str(max_length) + 's : %s'
    print('>>> 参数:')
    for keyPair in sorted(options.items()):
        print(fmt_string % keyPair)

    return options, model_obj, trainer_class, dataset_name, sub_data


def main():
    # 数据的文件始终在其父目录
    dataset_prefix = os.path.dirname(os.path.realpath(__file__))

    # 解析参数
    options, model_obj, trainer_class, dataset_name, sub_data = read_options()

    #
    is_leaf = options['data_format'] == 'json'
    if is_leaf:
        train_path = os.path.join(dataset_prefix, 'leaf', 'data', dataset_name, 'data', 'train')
        test_path = os.path.join(dataset_prefix, 'leaf', 'data', dataset_name, 'data', 'test')
    else:
        train_path = os.path.join(dataset_prefix, 'dataset', dataset_name, 'data', 'train')
        test_path = os.path.join(dataset_prefix, 'dataset', dataset_name, 'data', 'test')

    all_data_info = read_data(train_path, test_path, sub_data=sub_data, data_format=options['data_format'])
    # 调用solver
    trainer = trainer_class(options, model_obj, all_data_info)
    trainer.train()


if __name__ == '__main__':
    main()
