from torch.autograd import Variable
import torch
import torch.utils.data as ds


class Trainer:
    def __init__(self, net, data_set, optimizer, criterion, save='yolo.pt'):
        self.net = net
        self.data_loader = ds.DataLoader(data_set, shuffle=True)
        self.optimizer = optimizer
        self.criterion = criterion
        self.save = save

    def train(self):
        print('Training started!')
        self.net.train()
        l = 0
        for index, dic in enumerate(self.data_loader):
            image, label = Variable(dic['image'].float()), Variable(dic['label'].float())
            image = image.cuda()
            label = label.cuda()
            self.optimizer.zero_grad()
            output = self.net(image)
            loss = self.criterion(output, label)
            loss.backward()            
            self.optimizer.step()
            l+= loss
            if index % 10 == 0:
                print('Train : [{}/{} ({:.0f}%)]'.format(index * len(image),
                                                                       len(self.data_loader.dataset),
                                                                       100. * index / len(self.data_loader)))
                print('Avarage loss: ', l/(index+1))
        print('Training finished!')
        torch.save(self.net.state_dict(), self.save)
