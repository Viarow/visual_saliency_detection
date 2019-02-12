import torch
from torch.nn import utils
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torchvision import transforms
from dataset import DataMaskSet
from dataset_roots import MSRAB_PATH, MSRA10K_PATH
from loss import finalLoss, mean_abs_error
from visualize import Board
from network import VSDNet
from argparse import ArgumentParser


input_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
             ])
target_transform = transforms.Compose([
            transforms.Resize(256, 256),
            transforms.ToTensor(),
            # transforms.Lambda(lambda x: torch.round(x))
            ])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SELECT = [1, 2, 3, 6]

def train(args, model):
    trainSet = DataMaskSet(MSRAB_PATH, input_transform, target_transform)
    trainLoader = data.DataLoader(trainSet, args.batch_size, shuffle=True, num_workers=args.num_workers)
    model.train()
    optimizer = optim.Adam(model.parameters, args.lr)
    criterion = finalLoss().to(device)
    board = Board()

    for epoch in range(args.num_epochs):
        losses = []

        for i, (inputs, labels) in enumerate(trainLoader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(inputs)
            loss = criterion(preds, labels)
            losses.append(loss)
            loss.backward()
            optimizer.step()

    if (epoch+1) % args.show_period == 0:
        print('epoch: [%d/%d]   epoch_loss: [%.3f]' % (epoch, sum(losses)/len(losses)))
        board.show_image(inputs)
        board.show_image(preds)
        board.show_image(labels)

    if (epoch+1) % args.save_period == 0:
        filename = 'epoch{}.pth'.format(epoch+1)
        torch.save(model.state_dict(), filename)
        print('{} saved'.format(filename))


def precision_recall(self, pred, labels, step=100):
    precision, recall = torch.zeros(step), torch.zeros(step)
    axis = torch.linspace(0, 1.0, step)
    for i in range(step):
        scale = (pred >= axis[i]).float()
        precision[i] = (scale*labels).sum() / (scale + 1e-6)
        recall[i] = (scale*labels).sum() / (labels.sum())
    return precision, recall


def test(args, model):
    testSet = DataMaskSet(args.test_path, input_transform, target_transform)
    testLoader = data.DataLoader(testSet, args.batch_size, shuffle=1, num_workers=args.num_workers)
    model.eval()
    img_num = len(testSet)
    avg_mae = 0.0
    avg_prec, avg_recall = torch.zeros(args.step), torch.zeros(args.step)
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(testSet):
            # inputs, labels = inputs.unsqueeze(0), labels.unsqueeze(0)
            inputs, labels = inputs.to(device), labels.to(device)
            pred = model(inputs)
            pred_cat = torch.cat([pred[i] for i in SELECT], dim=1)
            pred = torch.mean(pred_cat, dim=1, keepdim=True)
            pred = F.interpolate(pred, labels.size()[2:], mode='bilinear', align_corners=True)
            mae = mean_abs_error(pred, labels)
            prec, recall = precision_recall(pred, labels,  args.step)
            avg_mae += mae
            avg_prec, avg_recall = avg_prec+ + prec, avg_recall + recall

    avg_mae = avg_mae / img_num
    avg_prec, avg_recall = avg_prec / img_num, avg_recall / img_num
    score = (1 + args.beta ** 2) * avg_prec * avg_recall / (args.beta **2 * avg_prec + avg_recall)
    score[score != score] = 0
    print('average mae: %.3f, max F-measure: %.3f' % (avg_mae, score.max()))


def main(args):

    net = VSDNet()
    net = net.to(device)

    if args.mode == 'train':
        train(args, net)
    elif args.mode == 'test':
        test(args, net)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('batch_size', type=int, default=1)
    parser.add_argument('num_workers', type=int, default=2)

    subparsers = parser.add_subparsers(dest='mode')
    subparsers.required = True

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('lr', type=float, default=1e-6)
    parser_train.add_argument('num_epochs', type=int, default=50)
    parser_train.add_argument('show_period', type=int, default=5)
    parser_train.add_argument('save_period', type=int, default=10)

    parser_test = subparsers.add_parser('test')
    parser_test.add_argument('test_path', required=True)
    parser_test.add_argument('step', type=int, default=100)
    parser_test.add_argument('beta', type=float, default=0.3)









