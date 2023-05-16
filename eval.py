import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
from thop import profile, clever_format
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import classification_report
import utils
from dataset_cifarLT import CIFAR10LT
from scheduler import LinearWarmupCosineAnnealingLR
from model import Model
import warnings
warnings.filterwarnings('ignore')

np.random.seed(1234)
torch.manual_seed(1234)

def knn_scikit(X_train, y_train, X_test, y_test, k):
    cls = KNN(n_neighbors=k) #, weights = 'distance')
    cls.fit(X_train, y_train)
    y_pred = cls.predict(X_test)
    return classification_report(y_test, y_pred, digits=4, output_dict=False)

def test(net, memory_data_loader, test_data_loader, temperature):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature, _ = net(data.cuda(non_blocking=True))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, _, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature, out = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR with Dual Temp Scheduling')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--batch_size', default=128, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--model_path', type=str, help='/path/to/model')
    parser.add_argument('--datapath', default='./datasets/', help='path to dataset')
    parser.add_argument('--dataset', default='cifar10', help='name of dataset')
    parser.add_argument('--lt', action='store_true', help='long-tailed or not')
    parser.add_argument('--ratio', default=0.01, type=float, help='Imbalance ratio -- only if imbal=True')

    # args parse
    args = parser.parse_args()
    feature_dim, batch_size = args.feature_dim, args.batch_size
    ds = 'BAL' if not args.lt else 'LT'

    # data prepare
    # if 'LT' in args.model_path:
    #     ds = 'LT'
    #     ratio = float(os.path.basename(args.model_path).split('LT[')[1].split(']')[0])
    #     assert ratio in [0.1, 0.01]

    if args.dataset == 'cifar10' and args.lt == True:
        print(f'>>> CIFAR-10-LT is being used with Imbalance ratio {args.ratio}')
        train_data = CIFAR10LT(root=args.datapath, train=True, transform=utils.train_transform, imb_factor=args.ratio)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        memory_data = CIFAR10LT(root=args.datapath, train=True, imb_factor=args.ratio, transform=utils.test_transform)
        memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_data = utils.CIFAR10Pair(root=args.datapath, train=False, transform=utils.test_transform, download=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
    elif args.dataset == 'cifar10' and args.lt == False:
        args.ratio = 1.0
        train_data = utils.CIFAR10Pair(root=args.datapath, train=True, transform=utils.train_transform, download=False)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,drop_last=True)
        memory_data = utils.CIFAR10Pair(root=args.datapath, train=True, transform=utils.test_transform, download=False)
        memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_data = utils.CIFAR10Pair(root=args.datapath, train=False, transform=utils.test_transform, download=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    elif args.dataset == 'cifar100' and args.lt == True:
        print(f'>>> CIFAR-100-LT is being used with Imbalance ratio {args.ratio}')
        train_data = CIFAR100LT(root=args.datapath, train=True, transform=utils.train_transform, imb_factor=args.ratio)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        memory_data = CIFAR100LT(root=args.datapath, train=True, imb_factor=args.ratio, transform=utils.test_transform)
        memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_data = utils.CIFAR100Pair(root=args.datapath, train=False, transform=utils.test_transform, download=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
    elif args.dataset == 'cifar100' and args.lt == False:
        args.ratio = 1.0
        train_data = utils.CIFAR100Pair(root=args.datapath, train=True, transform=utils.train_transform, download=False)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,drop_last=True)
        memory_data = utils.CIFAR100Pair(root=args.datapath, train=True, transform=utils.test_transform, download=False)
        memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_data = utils.CIFAR100Pair(root=args.datapath, train=False, transform=utils.test_transform, download=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # model setup and optimizer config
    model = Model(feature_dim)
    flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32),))
    flops, params = clever_format([flops, params])

    # load model weights
    model.load_state_dict(torch.load(args.model_path), strict=True)
    model = model.cuda()
    save_name_pre = os.path.basename(args.model_path).split('_model.pth')[0]

    c = len(memory_data.classes)

     # evaluation after full training
    model.eval()
    X_train, X_test, y_train, y_test = [],[],[],[]
    print(f">>> Evaluation of trained model: {save_name_pre}")
    with torch.no_grad():
        for data, _, target in tqdm(memory_loader, desc='Feature extracting'):
            feat, _ = model(data.cuda(non_blocking=True))
            X_train.append(feat)
            y_train.append(target.cuda(non_blocking=True))
        for data, _, target in tqdm(test_loader, desc='Feature extracting'):
            feat, _ = model(data.cuda(non_blocking=True))
            X_test.append(feat)
            y_test.append(target.cuda(non_blocking=True))
        
    X_train = torch.cat(X_train, dim=0).contiguous()
    X_train = X_train.cpu().numpy()
    y_train = torch.cat(y_train, dim=0).contiguous()
    y_train = y_train.cpu().numpy()

    X_test = torch.cat(X_test, dim=0).contiguous()
    X_test = X_test.cpu().numpy()
    y_test = torch.cat(y_test, dim=0).contiguous()
    y_test = y_test.cpu().numpy()

    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    uniformity = 0.0
    tolerance = 0.0

    for i in range(X_test.shape[0]):
        for j in range(X_test.shape[0]):
            if y_test[i]==y_test[j]:
                tolerance += np.multiply(X_test[i], X_test[j]).sum()
            # else:
            uniformity += np.exp(-2*np.sum(np.square(X_test[i] - X_test[j])))

    tolerance = tolerance/((X_test.shape[0]**2)/10)
    uniformity = np.log(uniformity/X_test.shape[0]**2)

    print('Tolerance:' ,tolerance, '|| Uniformity: ',uniformity)

    # print(knn_scikit(X_train, y_train, X_train, y_train, 20))
    
    outfile = open('results/{}_KNNeval.txt'.format(save_name_pre), 'w')
    outfile.write(os.path.basename(args.model_path) + "| CIFAR-10 " + ds + " | Balanced Test-Set")

    kvals = [1, 10] if ds == 'LT' else [10, 200]

    for k in kvals:
        outfile.write("\n------------------------------------")
        outfile.write("\nValue of K : " + str(k)+"\n") 
        cls_report = knn_scikit(X_train, y_train, X_test, y_test, k)
        print(cls_report)
        outfile.write(cls_report)
        outfile.write("------------------------------------\n\n")

    outfile.close()
    