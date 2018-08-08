import Dataset
import Loss
import YoloNet
import Trainer
import Tester
from torch import optim
import torch


def main():
    net = YoloNet.YOLONet()
    criterion = Loss.MyLoss()
    if input('Do you want to load network?').upper() == 'N':
        optimizer = optim.Adam(net.parameters(), lr=1e-4)
        train_data = Dataset.DetectionDataSet()
        trainer = Trainer.Trainer(net=net, data_set=train_data, optimizer=optimizer, criterion=criterion)
        trainer.train()
    else:
        net.load_state_dict(torch.load('yolo.pt'))
    test_data = Dataset.DetectionDataSet(paths='numpy_test/paths.txt', label_dir='numpy_test/', root_dir='test/')
    tester = Tester.Tester(net=net, test_criterion=criterion, data_set=test_data)
    tester.test()


if __name__ == '__main__':
    main()
