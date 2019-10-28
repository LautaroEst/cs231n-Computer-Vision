import torch
from torchvision import datasets as dset
from torch.utils.data import DataLoader, sampler
from torchvision.transforms import Compose, ToTensor


class Reshape(object):
    
    def __call__(self, sample):
        return sample[0,:,:]
        


def GetCIFAR10DataLoaders(NUM_TRAIN, NUM_VAL, batch_size=64):
    
    
    transform = Compose([ToTensor(),
                         Reshape()])
    
    mnist_train_val = dset.MNIST('./MNIST/train', train=True, download=True, transform=transform)
    mnist_test = dset.MNIST('./MNIST/test', train=False, download=True, transform=transform)
    
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