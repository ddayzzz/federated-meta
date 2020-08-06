import torch.utils.data as data
import os
import os.path
import errno
import numpy as np
import pickle
from PIL import Image
from torchvision import transforms

np.random.seed(6)


class Omniglot(data.Dataset):
    urls = [
        'https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip',
        'https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip'
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    '''
    The items are (filename,category). The index of all the categories can be found in self.idx_classes
    Args:
    - root: the directory where the dataset will be stored
    - transform: how to transform the input
    - target_transform: how to transform the target
    - download: need to download the dataset
    '''

    def __init__(self, root, transform=None, target_transform=None, download=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        if not self._check_exists():
            if download:
                self.download()
            else:
                raise RuntimeError('Dataset not found.' + ' You can use download=True to download it')
        # 图像的信息
        self.all_items = find_classes(os.path.join(self.root, self.processed_folder))
        # Dict: 图像的ID(由字符类别+对应其中的编号组成)->对应index
        self.idx_classes = index_classes(self.all_items)

    def __getitem__(self, index):

        filename = self.all_items[index][0]
        # 完整的文件路径
        img = str.join(os.sep, [self.all_items[index][2], filename])
        # 类型名称: 字符集名称/字符的类型名
        target = self.idx_classes[self.all_items[index][1]]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.all_items)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, "images_evaluation")) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, "images_background"))

    def download(self):
        """
        下载文件
        :return:
        """
        from six.moves import urllib
        import zipfile

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('== Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            file_processed = os.path.join(self.root, self.processed_folder)
            print("== Unzip from " + file_path + " to " + file_processed)
            zip_ref = zipfile.ZipFile(file_path, 'r')
            zip_ref.extractall(file_processed)
            zip_ref.close()
        print("Download finished.")


def find_classes(root_dir):
    retour = []
    for (root, dirs, files) in os.walk(root_dir):
        for f in files:
            # 遍历的文件名
            if f.endswith("png"):
                r = root.split(os.sep)
                lr = len(r)
                # 保存: (文件名, 字符集名称/字符的类型名(这里用character[xx]来表示), 文件存在的路径前缀)
                retour.append((f, r[lr - 2] + os.sep + r[lr - 1], root))
    print(">>> Found %d items " % len(retour))
    return retour


def index_classes(items):
    idx = {}
    for i in items:
        if i[1] not in idx:
            idx[i[1]] = len(idx)
    print(">>> Found %d classes" % len(idx))
    return idx


class OmniglotNShot:

    def __init__(self, root, imgsz):
        """
        设置 Omniglot 的数据集的格式
        :param root: 保存有 omniglot 数据集的路径 (e.g. dataset/omniglot)
        :param imgsz: 图像大小
        """
        self.imgsz = imgsz
        self.root = root
        # 加载数据
        self.x = self.load_data()
        self.x_train, self.x_test = self.x[:1200], self.x[1200:]

    def load_data(self):
        # if root/data.npy does not exist, just download it
        x = Omniglot(os.path.sep.join((self.root, 'data')), download=True,
                     transform=transforms.Compose([lambda x: Image.open(x).convert('L'),
                                                   lambda x: x.resize((self.imgsz, self.imgsz)),
                                                   lambda x: np.reshape(x, (self.imgsz, self.imgsz, 1)),
                                                   lambda x: np.transpose(x, [2, 0, 1]),
                                                   lambda x: x / 255.])
                     )

        temp = dict()  # {label:img1, img2..., 20 imgs, label2: img1, img2,... in total, 1623 label}
        # 这里按照类排序
        for (img, label) in x:
            if label in temp.keys():
                temp[label].append(img)
            else:
                temp[label] = [img]

        x = []
        for label, imgs in temp.items():  # labels info deserted , each label contains 20imgs
            x.append(np.array(imgs))

        # as different class may have different number of imgs
        # 每个类别只有20张图像
        x = np.array(x).astype(np.float)  # [[20 imgs],..., 1623 classes in total]
        # each character contains 20 imgs
        print('>>> Data shape:', x.shape)  # [1623, 20, 84, 84, 1]
        return x

    def normalization(self):
        """
        对于数据进行归一化
        :return:
        """
        self.mean = np.mean(self.x_train)
        self.std = np.std(self.x_train)
        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)
        # print("before norm:", "mean", self.mean, "max", self.max, "min", self.min, "std", self.std)
        self.x_train = (self.x_train - self.mean) / self.std
        self.x_test = (self.x_test - self.mean) / self.std

        self.mean = np.mean(self.x_train)
        self.std = np.std(self.x_train)
        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)

    def generate_dataset_for_fair(self, one_hot):
        """
        生成 ICLR 2020 fair learning 格式的数据集.
        :return:
        """
        # 对应的 task 拥有的数据的类的索引号: 前300个task数据范围为 [0, 1200); 后100个为 [1200, 1623), 注意这个是类
        # 后100个任务只有后 423 类型的数据, 这样造成了 task 的不同, 目前不清楚为什么
        assert self.x.shape[0] == 1623
        task_to_class = {}
        for i in range(400):  # 400 tasks
            if i < 300:  # first 300 meta-training tasks
                # TODO 看样子是 5 ways
                """
                从 numpy 的例子可以间接看出, choice 在不放回的情况下是无重复的 
                >>> np.random.choice(5, 3, replace=False)
                array([3,1,0])
                >>> #This is equivalent to np.random.permutation(np.arange(5))[:3]
                """
                class_ids = np.random.choice(1200, 5)
                task_to_class[i] = class_ids
            else:
                # 后面的 426
                class_ids = np.random.choice(range(1200, 1623), 5)  # 这样是没有重复的
                task_to_class[i] = class_ids
        # 拥有对应类的 task 的编号
        class_to_task = {}
        for i in range(1623):
            class_to_task[i] = []
        for i in range(400):
            for j in task_to_class[i]:
                class_to_task[j].append(i)

        X_test = {}  # testing test of all tasks (300 meta-train + 100 meta-test)
        y_test = {}
        X_train = {}  # training set of all tasks (300 meta-train + 100 meta-test)
        y_train = {}

        for i in range(400):
            X_test[i] = []
            y_test[i] = []
            X_train[i] = []
            y_train[i] = []

        all_data = []
        for idx in range(self.x.shape[0]):
            # idx 代表的 class id
            for i in range(self.x.shape[1]):
                # i 指定对应的 shot
                content = self.x[idx, i, :, :, :]
                # 取出的图像为 28 * 28 = 784 (不考虑最后一维作为灰度)
                content = content.flatten()
                all_data.append(content)
                # x.shape[1] == 20, < 10 的部划分给 train, >= 10 的部分给 test
                if i < 10:
                    for device_id in class_to_task[idx]:
                        X_train[device_id].append(content)
                        # np.where 找到对应的位置. 这个位置的取值为 [0, 5) 的整数. idx 一定是存在的, np.where 返回的就是idx在其中的坐标(注意只有5个类)
                        y_train[device_id].append(int(np.where(task_to_class[device_id] == idx)[0][0].astype(np.int32)))

                else:
                    for device_id in class_to_task[idx]:
                        X_test[device_id].append(content)
                        y_test[device_id].append(int(np.where(task_to_class[device_id] == idx)[0][0].astype(np.int32)))

        all_data = np.asarray(all_data)
        print(">>> original data:", all_data[0])
        print('>>> Y[100]:', y_train[100])
        print('>>> Y[399]:', y_train[399])
        # some simple normalization
        mu = np.mean(all_data.astype(np.float32), 0)
        print(">>> mu:", mu)
        sigma = np.std(all_data.astype(np.float32), 0)

        for device_id in range(400):
            X_train[device_id] = np.array(X_train[device_id])
            X_test[device_id] = np.array(X_test[device_id])

        for device_id in range(400):
            X_train[device_id] = (X_train[device_id].astype(np.float32) - mu) / (sigma + 0.001)
            X_test[device_id] = (X_test[device_id].astype(np.float32) - mu) / (sigma + 0.001)
            X_train[device_id] = X_train[device_id].tolist()
            X_test[device_id] = X_test[device_id].tolist()
        # 生成数据
        train_file = os.sep.join((self.root, 'data', 'train', 'fair_task[400].pkl'))
        test_file = os.sep.join((self.root, 'data', 'test', 'fair_task[400].pkl'))
        num_device = 400  # device is task
        train_data = {'users': [], 'user_data': {}, 'num_samples': []}
        test_data = {'users': [], 'user_data': {}, 'num_samples': []}
        for i in range(num_device):
            uname = "class_" + str(i)
            # users 是按照顺序添加的, 因此不会造成顺序的问题
            train_data['users'].append(uname)
            train_data['user_data'][uname] = {'x': X_train[i], 'y': y_train[i]}
            test_data['users'].append(uname)
            test_data['user_data'][uname] = {'x': X_test[i], 'y': y_test[i]}


        with open(train_file, 'wb') as outfile:
            pickle.dump(train_data, outfile, pickle.HIGHEST_PROTOCOL)
        with open(test_file, 'wb') as outfile:
            pickle.dump(test_data, outfile, pickle.HIGHEST_PROTOCOL)

    def generate_dataset(self, data_pack, num_tasks):
        """
        Collects several batches data for N-shot learning
        :param data_pack: [cls_num, 20, 84, 84, 1]
        :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
        """
        n_way = 5
        support_shots = 10
        query_shots = 10
        #  take 5 way 1 shot as example: 5 * 1
        setsz = support_shots * n_way
        querysz = query_shots * n_way
        data_cache = []

        x_spts, y_spts, x_qrys, y_qrys = [], [], [], []
        for i in range(num_tasks):  # one batch means one task

            x_spt, y_spt, x_qry, y_qry = [], [], [], []
            selected_cls = np.random.choice(data_pack.shape[0], n_way, False)

            for j, cur_class in enumerate(selected_cls):
                selected_img = np.random.choice(20, support_shots + query_shots, False)

                # meta-training and meta-test
                x_spt.append(data_pack[cur_class][selected_img[:support_shots]])
                x_qry.append(data_pack[cur_class][selected_img[support_shots:]])
                y_spt.append([j for _ in range(support_shots)])
                y_qry.append([j for _ in range(query_shots)])

            # shuffle, 同时 flatten
            perm = np.random.permutation(n_way * support_shots)
            x_spt = np.array(x_spt).reshape([n_way * support_shots, self.imgsz * self.imgsz])[perm]
            y_spt = np.array(y_spt).reshape([n_way * support_shots])[perm]
            perm = np.random.permutation(n_way * query_shots)
            x_qry = np.array(x_qry).reshape([n_way * query_shots, self.imgsz * self.imgsz])[perm]
            y_qry = np.array(y_qry).reshape([n_way * query_shots])[perm]

            x_spts.append(x_spt)
            y_spts.append(y_spt)
            x_qrys.append(x_qry)
            y_qrys.append(y_qry)

        # [num_tasks, setsz, 1, 84, 84]
        x_spts = np.array(x_spts).astype(np.float32).reshape([num_tasks, setsz, self.imgsz * self.imgsz])
        y_spts = np.array(y_spts).astype(np.int).reshape([num_tasks, setsz])
        # [num_tasks, qrysz, 1, 84, 84]
        x_qrys = np.array(x_qrys).astype(np.float32).reshape([num_tasks, querysz, self.imgsz * self.imgsz])
        y_qrys = np.array(y_qrys).astype(np.int).reshape([num_tasks, querysz])
        return (x_spts, y_spts, x_qrys, y_qrys)

    def generate_dataset_for_MAML_in_fl(self, one_hot):
        self.normalization()
        (train_x_spts, train_y_spts, train_x_qrys, train_y_qrys) = self.generate_dataset(data_pack=self.x_train, num_tasks=300)
        (test_x_spts, test_y_spts, test_x_qrys, test_y_qrys) = self.generate_dataset(data_pack=self.x_test, num_tasks=100)
        # 生成数据
        train_file = os.sep.join((self.root, 'data', 'train', 'fair_task[400]_1.pkl'))
        test_file = os.sep.join((self.root, 'data', 'test', 'fair_task[400]_1.pkl'))

        train_data = {'users': [], 'user_data': {}, 'num_samples': []}
        test_data = {'users': [], 'user_data': {}, 'num_samples': []}
        # meta-train
        for i in range(300):
            uname = "class_" + str(i)
            # users 是按照顺序添加的, 因此不会造成顺序的问题
            train_data['users'].append(uname)
            train_data['user_data'][uname] = {'x': train_x_spts[i], 'y': train_y_spts[i]}
            test_data['users'].append(uname)
            test_data['user_data'][uname] = {'x': train_x_qrys[i], 'y': train_y_qrys[i]}
        # meta-test
        for i in range(100):
            uname = "class_" + str(i + 300)
            # users 是按照顺序添加的, 因此不会造成顺序的问题
            train_data['users'].append(uname)
            train_data['user_data'][uname] = {'x': test_x_spts[i], 'y': test_y_spts[i]}
            test_data['users'].append(uname)
            test_data['user_data'][uname] = {'x': test_x_qrys[i], 'y': test_y_qrys[i]}

        with open(train_file, 'wb') as outfile:
            pickle.dump(train_data, outfile, pickle.HIGHEST_PROTOCOL)
        with open(test_file, 'wb') as outfile:
            pickle.dump(test_data, outfile, pickle.HIGHEST_PROTOCOL)


def load_data():
    train_file = os.sep.join(('.', 'data', 'train', 'fair_task[400]_1.pkl'))
    test_file = os.sep.join(('.', 'data', 'test', 'fair_task[400]_1.pkl'))
    with open(train_file, 'rb') as fp:
        train = pickle.load(fp)
    with open(test_file, 'rb') as fp:
        test = pickle.load(fp)
    return train, test


def test():
    train, test = load_data()
    train_samples_size = []
    test_samples_size = []
    for i in range(400):
        uname = 'class_' + str(i)
        train_x = train['user_data'][uname]['x']
        train_y = train['user_data'][uname]['y']
        test_x = test['user_data'][uname]['x']
        test_y = test['user_data'][uname]['y']
        # train 和 test 确实不同, 但是生成的文件确实相同很奇怪
        train_samples_size.append(len(train_y))
        test_samples_size.append(len(test_y))
    train_mean = np.mean(train_samples_size)
    train_var = np.var(train_samples_size)
    test_mean = np.mean(test_samples_size)
    test_var = np.var(test_samples_size)
    print(train_mean, train_var, test_mean, test_var)

if __name__ == '__main__':
    prefix = os.path.dirname(__file__)
    if len(prefix) <= 0:
        prefix = '.'
    paths = ['{prefix}{sep}data', '{prefix}{sep}data{sep}train', '{prefix}{sep}data{sep}test']
    for path in paths:
        p = path.format(sep=os.sep, prefix=prefix)
        if not os.path.exists(p):
            os.mkdir(p)
    omniglot_gen = OmniglotNShot(root=prefix, imgsz=28)
    # omniglot_gen.generate_dataset_for_fair()
    omniglot_gen.generate_dataset_for_MAML_in_fl(one_hot=False)
    # test()
