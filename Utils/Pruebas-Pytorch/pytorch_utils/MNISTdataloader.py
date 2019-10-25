import torch
from torchvision import datasets as dset
from torch.utils.data import DataLoader, sampler
from torchvision.transforms import ToTensor


def GetMNISTDataLoaders(NUM_TRAIN, NUM_VAL, batch_size=64):
    
    mnist_train_val = dset.MNIST('./MNIST/train', train=True, download=True, transform=ToTensor())
    mnist_test = dset.MNIST('./MNIST/test', train=False, download=True, transform=ToTensor())
    
    print('La base de datos MNIST contiene', len(mnist_train_val) + len(mnist_test), 'muestras')
    print('Cantidad de muestras para entrenamiento: ', NUM_TRAIN)
    print('Cantidad de muestras para validación: ', NUM_VAL)
    print('Cantidad de muestras para test: ', len(mnist_test))
    
    mnist_train_dataloader = DataLoader(mnist_train_val, 
                                        batch_size=batch_size, 
                                        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))
    mnist_val_dataloader = DataLoader(mnist_train_val, 
                                      batch_size=batch_size, 
                                      sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN,NUM_TRAIN+NUM_VAL)))
    mnist_test_dataloader = DataLoader(mnist_test, 
                                      batch_size=batch_size)  
    
    return mnist_train_dataloader, mnist_val_dataloader, mnist_test_dataloader


