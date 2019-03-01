#coding:utf-8
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from cnnNet import cnnNet

if __name__ == '__main__':
    path = os.path.abspath('.')
    path_train = os.path.join(path,'data/original_data')
    path_para = os.path.join(path,'parameters/model.pt')

    #1,download data
    transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    trainset = torchvision.datasets.CIFAR10(root=path_train, train=True, download=False,
                                            transform = transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    #2,bulid model and use cuda
    cnnNet = cnnNet()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("We use the device '{}' to train our model".format(device))
    cnnNet.to(device)

    #3, optimizer and loss
    criterion =nn.CrossEntropyLoss()
    optimizer = optim.SGD(cnnNet.parameters(),lr=0.001, momentum=0.9)

    #4,print the parameters of cnnNet and optimizer
    print("Model's state_dict: ")
    for param_tensor in cnnNet.state_dict():
        print(param_tensor, "\t", cnnNet.state_dict()[param_tensor].size())
    print("parameter's state_dict: ")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

    #5,start training
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader,0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = cnnNet(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/2000))
                running_loss = 0.0
    print('Finished Training and Started saving model')
    #6，save model
    torch.save(cnnNet.state_dict(),path_para)
    print('END!!!')





