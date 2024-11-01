from torchvision.datasets import MNIST


def test_mnist_downloading():
    data = MNIST("./datasets", train=True, download=True)
    print(data)
