import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from .utils_data import filter_class






def mnist2d_10class(data_dir):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = datasets.MNIST(data_dir, train=True, transform=transform, download=True)
    testset = datasets.MNIST(data_dir, train=False, transform=transform, download=True)
    num_classes = 10
    return trainset, testset, num_classes

def mnist2d_5class(data_dir):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = datasets.MNIST(data_dir, train=True, transform=transform, download=True)
    testset = datasets.MNIST(data_dir, train=False, transform=transform, download=True)
    trainset, num_classes = filter_class(trainset, [0, 1, 2, 3, 4])
    testset, _ = filter_class(testset, [0, 1, 2, 3, 4])
    num_classes = 5
    return trainset, testset, num_classes

def mnist2d_2class(data_dir):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = datasets.MNIST(data_dir, train=True, transform=transform, download=True)
    testset = datasets.MNIST(data_dir, train=False, transform=transform, download=True)
    trainset, num_classes = filter_class(trainset, [0, 1])
    testset, _ = filter_class(testset, [0, 1])
    return trainset, testset, num_classes

def mnistvector_10class(data_dir):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.flatten())
    ])
    trainset = datasets.MNIST(data_dir, train=True, transform=transform, download=True)
    testset = datasets.MNIST(data_dir, train=False, transform=transform, download=True)
    num_classes = 10
    return trainset, testset, num_classes

def mnistvector_5class(data_dir):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.flatten())
    ])
    trainset = datasets.MNIST(data_dir, train=True, transform=transform, download=True)
    testset = datasets.MNIST(data_dir, train=False, transform=transform, download=True)
    trainset, num_classes = filter_class(trainset, [0, 1, 2, 3, 4])
    testset, _ = filter_class(testset, [0, 1, 2, 3, 4])
    return trainset, testset, num_classes

def mnistvector_2class(data_dir):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.flatten())
    ])
    trainset = datasets.MNIST(data_dir, train=True, transform=transform, download=True)
    testset = datasets.MNIST(data_dir, train=False, transform=transform, download=True)
    trainset, num_classes = filter_class(trainset, [0, 1])
    testset, _ = filter_class(testset, [0, 1])
    return trainset, testset, num_classes


if __name__ == '__main__':
    trainset, testset, num_classes = mnist2d_2class('./data/')
    trainloader  = DataLoader(trainset, batch_size=trainset.data.shape[0])
    print(trainset)
    print(testset)
    print(num_classes)

    batch_imgs, batch_lbls = next(iter(trainloader))
    print(batch_imgs.shape, batch_lbls.shape)
    print(batch_lbls.unique(return_counts=True))
