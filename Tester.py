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
        score = {'hit': 0, 'miss': 0, 'miss_class': 0, 'hit_class': 0}
        for index, dic in enumerate(self.data_loader):
            data, target = Variable(dic['image'].float()), Variable(dic['label'].float())
            with torch.no_grad():
                output = self.net(data)
                test_loss += self.criterion(output, target)
                img_score = Util.score(data, target, output, 'anc_predictions', index=index)
                score['hit'] += img_score['hit']
                score['miss'] += img_score['miss']
                score['miss_class'] += img_score['miss_class']
            test_loss /= len(self.data_loader.dataset)
            boxes = 0
            for key in score.keys():
                boxes += score[key]
            for key in score.keys():
                score[key] *= 100 / boxes
            print("Average loss:", test_loss)
            print('Score: ')
            print('{0:.2f}% of numbers located'.format(score['hit'] + score['miss_class']))
            print('{0:.2f}% of numbers correctly classified'.format(score['hit'] + score['hit_class']))
            print('{0:.2f}% of numbers located and correctly classified'.format(score['hit']))
            print('{0:.2f}% of predictions are wrong'.format(score['miss']))
            return test_loss


class AnchorTester:
    def __init__(self, net, data_set, test_criterion):
        self.net = net
        self.data_loader = ds.DataLoader(data_set, shuffle=True)
        self.criterion = test_criterion

    def test(self):
        print('Testing started!')
        self.net.eval()
        test_loss = 0
        score = {'hit': 0, 'miss': 0, 'miss_class': 0, 'hit_class': 0}
        for index, dic in enumerate(self.data_loader):
            data, target = Variable(dic['image'].float()), Variable(dic['label'].float())
            with torch.no_grad():
                output = self.net(data)
                test_loss += self.criterion(output, target)
                img_score = Util.anc_score(data, target, output, 'anc_predictions', index=index)
                score['hit'] += img_score['hit']
                score['miss'] += img_score['miss']
                score['miss_class'] += img_score['miss_class']
                score['hit_class'] += img_score['hit_class']
        test_loss /= len(self.data_loader.dataset)
        boxes = 0
        for key in score.keys():
            boxes += score[key]
        for key in score.keys():
            score[key] *= 100 / boxes
        print("Average loss:", test_loss)
        print('Score: ')
        print('{0:.2f}% of numbers located'.format(score['hit'] + score['miss_class']))
        print('{0:.2f}% of numbers correctly classified'.format(score['hit'] + score['hit_class']))
        print('{0:.2f}% of numbers located and correctly classified'.format(score['hit']))
        print('{0:.2f}% of predictions are wrong'.format(score['miss']))
        return test_loss
