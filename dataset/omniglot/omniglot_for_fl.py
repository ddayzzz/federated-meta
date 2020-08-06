import os
from torchmeta.datasets.omniglot import Omniglot
from torchvision import transforms
from torchmeta.datasets.helpers import omniglot

root_path = os.path.dirname(os.path.realpath(__file__))


def get_omniglot_for_fl():
    trans = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
    ])
    meta_train_ds = Omniglot(root_path, num_classes_per_task=5, meta_train=True,
                             use_vinyals_split=False, download=True, transform=trans)
    # meta_test_ds = Omniglot(root_path, num_classes_per_task=5, meta_test=True,
    #                          use_vinyals_split=False, download=True, transform=trans)
    return (meta_train_ds, meta_test_ds)


if __name__ == '__main__':
    # (meta_train_ds, meta_test_ds) = get_omniglot_for_fl()
    from torchmeta.datasets.helpers import omniglot
    from torchmeta.utils.data import BatchMetaDataLoader

    dataset = omniglot(root_path, ways=5, shots=5, test_shots=15, meta_train=True, download=True)
    dataloader = BatchMetaDataLoader(dataset, batch_size=16, num_workers=4)

    for batch in dataloader:
        train_inputs, train_targets = batch["train"]
        print('Train inputs shape: {0}'.format(train_inputs.shape))  # (16, 25, 1, 28, 28)
        print('Train targets shape: {0}'.format(train_targets.shape))  # (16, 25)

        test_inputs, test_targets = batch["test"]
        print('Test inputs shape: {0}'.format(test_inputs.shape))  # (16, 75, 1, 28, 28)
        print('Test targets shape: {0}'.format(test_targets.shape))  # (16, 75)
    g = 5
