# GLOBAL PARAMETERS
import argparse
DATASETS = [ 'shakespeare', 'omniglot', 'femnist']
TRAINERS = {'fedmeta': 'FedMeta',
            'fedavg_adv': 'FedAvgAdv'}

OPTIMIZERS = TRAINERS.keys()
MODEL_CONFIG = {
    'mnist.logistic': {'out_dim': 10, 'in_dim': 784},
    'femnist.cnn': {'num_classes': 62, 'image_size': 28},
    'omniglot.cnn': {'num_classes': 5, 'image_size': 28},
    'shakespeare.stacked_lstm': {'seq_len': 80, 'num_classes': 80, 'num_hidden': 256, }
}


def base_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('--algo',
                        help='name of trainer;',
                        type=str,
                        choices=OPTIMIZERS,
                        default='fedavg')
    parser.add_argument('--dataset',
                        help='name of dataset;',
                        type=str,
                        required=True)
    parser.add_argument('--model',
                        help='name of model;',
                        type=str,
                        default='logistic')
    parser.add_argument('--wd',
                        help='weight decay parameter;',
                        type=float,
                        default=0.001)
    parser.add_argument('--device',
                        help='device',
                        default='cpu:0',
                        type=str)
    parser.add_argument('--num_rounds',
                        help='number of rounds to simulate;',
                        type=int,
                        default=200)
    parser.add_argument('--eval_on_test_every',
                        help='evaluate every ____ rounds;',
                        type=int,
                        default=1)
    parser.add_argument('--eval_on_train_every',
                        help='evaluate every ____ rounds;',
                        type=int,
                        default=1)
    parser.add_argument('--eval_on_validation_every',
                        help='evaluate every ____ rounds;',
                        type=int,
                        default=1)
    parser.add_argument('--save_every',
                        help='save global model every ____ rounds;',
                        type=int,
                        default=50)
    parser.add_argument('--clients_per_round',
                        help='number of clients trained per round;',
                        type=int,
                        default=10)
    parser.add_argument('--batch_size',
                        help='batch size when clients train on data;',
                        type=int,
                        default=10)
    parser.add_argument('--num_epochs',
                        help='number of epochs when clients train on data;',
                        type=int,
                        default=20)
    parser.add_argument('--lr',
                        help='learning rate for inner solver;',
                        type=float,
                        default=0.01)
    parser.add_argument('--seed',
                        help='seed for randomness;',
                        type=int,
                        default=0)
    parser.add_argument('--quiet',
                        help='仅仅显示结果的代码',
                        type=int,
                        default=0)
    parser.add_argument('--result_prefix',
                        help='保存结果的前缀路径',
                        type=str,
                        default='./result')
    parser.add_argument('--train_val_test',
                        help='数据集是否以训练集、验证集和测试集的方式存在',
                        action='store_true')
    parser.add_argument('--result_dir',
                        help='指定已经保存结果目录, 可以加载相关的 checkpoints',
                        type=str,
                        default='')
    # TODO 以后支持 之家加载 leaf 目录里的数据
    parser.add_argument('--data_format',
                        help='加载的数据格式, json 为 Leaf以及Li T.等人定义的格式, 默认为 pkl',
                        type=str,
                        default='pkl')
    parser.add_argument('--train_inner_step', default=0, type=int)
    parser.add_argument('--test_inner_step', default=0, type=int)
    parser.add_argument('--same_mini_batch', action='store_true', default=False)
    return parser


def add_dynamic_options(argparser):
    # 获取对应的 solver 的名称
    params = argparser.parse_known_args()[0]
    algo = params.algo
    # if algo in ['maml']:
    #     argparser.add_argument('--q_coef', help='q', type=float, default=0.0)
    if algo in ['fedmeta']:
        argparser.add_argument('--meta_algo', help='使用的元学习算法, 默认 maml', type=str, default='maml',
                               choices=['maml', 'reptile', 'meta_sgd'])
        argparser.add_argument('--outer_lr', help='更新元学习中的外部学习率', type=float, required=True)
        argparser.add_argument('--meta_train_test_split', type=int, default=-1)
        argparser.add_argument('--store_to_cpu', action='store_true', default=False)
    elif algo == 'fedavg_adv':
        argparser.add_argument('--use_all_data', action='store_true', default=False)

    return argparser
