from torch.autograd import Variable
import torch
import torch.utils.data as ds


class Trainer:
    def __init__(self, net, data_set, optimizer, criterion):
        self.net = net
        self.data_loader = ds.DataLoader(data_set, shuffle=True)
        self.optimizer = optimizer
        self.criterion = criterion

    def train(self):
        print('Training started!')
        self.net.train()
        for index, dic in enumerate(self.data_loader):
            image, label = Variable(dic['image'].float()), Variable(dic['label'].float())
            self.optimizer.zero_grad()
            output = self.net(image)
            loss = self.criterion(output, label)
            loss.backward()
            self.optimizer.step()
            if index % 10 == 0:
                print('Train : [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(index * len(image),
                                                                       len(self.data_loader.dataset),
                                                                       100. * index / len(self.data_loader),
                                                                       loss.item()))
        print('Training finished!')
        torch.save(self.net.state_dict(), 'yolo.pt')