#coding:utf-8
import os
import torch
import torchvision
import torchvision.transforms as transforms
from cnnNet import cnnNet

if __name__ == '__main__':
    path = os.path.abspath('.')
    path_test = os.path.join(path,'data/original_data')
    path_para = os.path.join(path,'parameters/model.pt')

    #1,download data
    transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    testset = torchvision.datasets.CIFAR10(root=path_test, train=False, download=False,
                                       transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)

    #2, load model
    cnnNet = cnnNet()
    cnnNet.load_state_dict(torch.load(path_para))
    cnnNet.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("We use the device '{}' to test".format(device))
    cnnNet.to(device)

    #3, starting test
    correct = 0
    total = 0
    # with torch.no_grad()表示改行以下代码中的参数不再需要梯度
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = cnnNet(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
