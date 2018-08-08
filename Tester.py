from torch.autograd import Variable
import torch
import Util
import torch.utils.data as ds


class Tester:
    def __init__(self, net, data_set, test_criterion):
        self.net = net
        self.data_loader = ds.DataLoader(data_set, shuffle=True)
        self.criterion = test_criterion

    def test(self):
        print('Testing started!')
        self.net.eval()
        test_loss = 0
        for index, dic in enumerate(self.data_loader):
            data, target = Variable(dic['image'].float()), Variable(dic['label'].float())
            with torch.no_grad():
                output = self.net(data)
                test_loss += self.criterion(output, target)
                Util.toImage(data, target, 'predictions/ground truth', index=index)
                Util.toImage(data, output, 'predictions/outputs', index=index)
        test_loss /= len(self.data_loader.dataset)
        print("Average loss:", test_loss)
