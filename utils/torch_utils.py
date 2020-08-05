import torch
import numpy as np


def zeros(shape, gpu=True, **kwargs):
    """
    得到全零的张量 TODO: 默认创建在GPU上的张量
    :param shape: 形状
    :param gpu: 是否使用gpu
    :param kwargs:
    :return:
    """
    return torch.zeros(*shape, **kwargs).cuda() if gpu else torch.zeros(*shape, **kwargs)


def get_flat_params_from(model):
    """
    得到扁平的模型的参数
    :param model: 模型类
    :return: 扁平化的参数，顺序为 model.parameters() 的顺序
    """
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params


def model_parameters_shape_list(model):
    """
    模型参数的形状
    :param model: 模型
    :return: 大小
    """
    return [x.size() for x in model.parameters()]


def from_flatten_to_parameter(shape_info, flat_params):
    """
    flat的参数->原始的形状
    :param shape_info: 维度信息(可由model_parameters_shape_list得出)
    :param flat_params: 扁平的参数
    :return:
    """
    new_params = []
    prev_ind = 0
    for shape in shape_info:
        # 计算 flat 后的乘积
        flat_size = int(np.prod(list(shape)))
        # 恢复值
        new_params.append(flat_params[prev_ind:prev_ind + flat_size].view(shape))
        prev_ind += flat_size
    return new_params


def set_flat_params_to(model, flat_params):
    """
    扁平化的参数转换为
    :param model:
    :param flat_params:
    :return:
    """
    prev_ind = 0
    for param in model.parameters():
        # 计算 flat 后的乘积
        flat_size = int(np.prod(list(param.size())))
        # 恢复值
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def get_grad_dict(output, inputs, filter_input_ids=set(), retain_graph=False, create_graph=False):
    """
    获得梯度信息
    :param output: 输出,一般是loss
    :param inputs: 模型的参数
    :param filter_input_ids: 去掉那些的不用的参数
    :param retain_graph:  retain_graph 和 create_graph 的值一般相同; 前者为 False 表示计算后销毁计算图
    :param create_graph:
    :return:
    """
    if create_graph:
        retain_graph = True

    inputs = list(inputs)
    params = []
    for i, param in enumerate(inputs):
        if i not in filter_input_ids:
            params.append(param)

    grads = torch.autograd.grad(output, params, retain_graph=retain_graph, create_graph=create_graph)

    j = 0
    out_grads = dict()
    for i, param in enumerate(inputs):
        if i in filter_input_ids:
            out_grads[i] = zeros(param.data.view(-1).shape)
        else:
            out_grads[i] = grads[j]
            j += 1

    for param in params:
        param.grad = None
    return out_grads


def get_flat_grad(output, inputs, filter_input_ids=set(), retain_graph=False, create_graph=False):
    if create_graph:
        retain_graph = True

    inputs = list(inputs)
    params = []
    for i, param in enumerate(inputs):
        if i not in filter_input_ids:
            params.append(param)

    grads = torch.autograd.grad(output, params, retain_graph=retain_graph, create_graph=create_graph)

    j = 0
    out_grads = []
    for i, param in enumerate(inputs):
        if i in filter_input_ids:
            out_grads.append(zeros(param.data.view(-1).shape))
        else:
            out_grads.append(grads[j].view(-1))
            j += 1
    grads = torch.cat(out_grads)

    for param in params:
        param.grad = None
    return grads


def get_flat_grad_from_sparse(output, inputs, filter_input_ids=set(), retain_graph=False, create_graph=False):
    """
    用于 Shakespeare 数据集
    :param output:
    :param inputs:
    :param filter_input_ids:
    :param retain_graph:
    :param create_graph:
    :return:
    """
    if create_graph:
        retain_graph = True

    inputs = list(inputs)
    params = []
    for i, param in enumerate(inputs):
        if i not in filter_input_ids:
            params.append(param)

    grads = torch.autograd.grad(output, params, retain_graph=retain_graph, create_graph=create_graph)

    j = 0
    out_grads = []
    for i, param in enumerate(inputs):
        if i in filter_input_ids:
            out_grads.append(zeros(param.data.view(-1).shape))
        else:
            out_grads.append(grads[j].view(-1))
            j += 1
    grads = torch.cat(out_grads)

    for param in params:
        # 去掉后续可能会计入图的操作
        param.grad = None
    # 开始处理(这里的 sparse 似乎和 embedding 有关, embedding 的 sparse 暂时为False, embedding 的梯度形状 [seq_lem, embedding_dim])
    # 这里的 grad0 是来自于 https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/IndexedSlices
    # 一个解释: https://www.zhihu.com/question/277403551
    # torch sparse: https://pytorch.org/docs/stable/sparse.html
    # 目前不使用 sparse
    # 的元素, 可能是 embedding 的缘故
    # dense[slices.indices[i], :, :, :, ...] = slices.values[i, :, :, :, ...], sliced 即为 grads[0]
    # indices = grads[0].indices
    # values = grads[0].values
    # first_layer_dense = np.zeros((80, 8))
    # for i in range(indices.shape[0]):
    #     first_layer_dense[indices[i], :] = values[i, :]
    # # 其他的梯度
    # client_grads = first_layer_dense
    # for i in range(1, len(grads)):
    #     client_grads = np.append(client_grads, grads[i])  # output a flattened array
    return grads