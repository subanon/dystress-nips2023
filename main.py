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

import utils
from dataset_cifarLT import CIFAR10LT, CIFAR100LT
from scheduler import LinearWarmupCosineAnnealingLR
from model import Model, Model2
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR

import datetime

np.random.seed(1234)
torch.manual_seed(1234)

#### Proposed Dual Temp Scheduling ####
def get_temp_positive(tmin, tmax, cossim):

    ###### UNCOMMENT FOR STABILIZATION ###########
    # tmin = tmin + global_step*(tmax-tmin)/epochs
    ##############################################

    if func == 'cossin':
        temp = tmin + 0.5*(tmax - tmin) *(1 + torch.cos((1 + cossim)*torch.tensor(np.pi)))  # sine-cosine scheduling  
    elif func=='cosconst':
        a, b, c = (float(x) for x in exp_params.split('|'))
        temp = torch.ones_like(cossim)
        temp[cossim > -0.2] = tmax #a - b * torch.exp(-c*cossim[cossim>0.0])
        temp[cossim <= -0.2] = tmin + 0.5*(tmax-tmin)*(1+torch.cos((0.2+cossim[cossim<=-0.2])*torch.tensor(np.pi)/0.8))
    else:
        temp = tmax
    return temp   


def get_temp_negative(tmin, tmax, cossim):
    
    ###### UNCOMMENT FOR STABILIZATION ###########
    # tmin = tmin + global_step*(tmax-tmin)/epochs
    ##############################################
    
    if func == 'cossin':
        temp = tmin + 0.5*(tmax - tmin) *(1 + torch.cos((1 + cossim)*torch.tensor(np.pi)))  # sine-cosine scheduling
    elif func=='cosconst':
        a, b, c = (float(x) for x in exp_params.split('|'))
        temp = torch.ones_like(cossim)
        temp[cossim > -0.2] = tmax #a - b * torch.exp(-c*cossim[cossim>0.0])
        temp[cossim <= -0.2] = tmin + 0.5*(tmax-tmin)*(1+torch.cos((0.2+cossim[cossim<=-0.2])*torch.tensor(np.pi)/0.8))
    else:
        temp = tmax
    return temp

def get_mask_negative(batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

# train for one epoch to learn unique features
def train(net, data_loader, train_optimizer, bsz, tmin, tmax):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for pos_1, pos_2, target in train_bar:
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
        _, z_1 = net(pos_1)
        _, z_2 = net(pos_2)
        # z_1 = torch.nn.functional.normalize(z_1, dim = 1, p = 2)
        # z_2 = torch.nn.functional.normalize(z_2, dim = 1, p = 2)
        
        N = 2 * bsz
        z = torch.cat([z_1, z_2], dim=0) # [2*B, D]

        # compute similarity matrix
        sim = z @ z.T
        sim_i_j = torch.diag(sim, bsz)
        sim_j_i = torch.diag(sim, -bsz)

        # get positive and negative pairs
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[precomp_mask].reshape(N, -1)

        '''Dual temp scheduling'''
        pos_temp, neg_temp = get_temp_positive(tmin, tmax, positive_samples.detach()), get_temp_negative(tmin, tmax, negative_samples.detach())
        positive_samples = positive_samples / pos_temp
        negative_samples = negative_samples / neg_temp
        
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = F.cross_entropy(logits, labels)

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num


# test for one epoch, use weighted knn(t) to find the most similar images' label to assign the test image
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
    parser.add_argument('--tmin', default=0.07, type=float, help='Min. Temperature used in temp scheduling')
    parser.add_argument('--tmax', default=0.2, type=float, help='Max. Temperature used in temp scheduling')
    parser.add_argument('--func', default='expo', type=str, choices=['cossin', 'const', 'cosconst'], help='temperature function to be used')
    parser.add_argument('--exp_params', default='0.2|0.1|3.5', type=str, help='parameters of exponential scheduling')
    parser.add_argument('--knn_t', default=0.1, type=float, help='Temperature value for KNN eval')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=128, type=int, help='Number of images in each mini-batch') #128
    parser.add_argument('--epochs', default=200, type=int, help='Number of sweeps over the dataset to train') # 200
    parser.add_argument('--opti', default='SGD', type=str, choices=['Adam', 'SGD'], help='optimizer chosen')
    parser.add_argument('--lr', default=0.2, type=float, help='learning rate chosen') #0.03 
    parser.add_argument('--datapath', default='./datasets/', help='path to dataset')
    parser.add_argument('--dataset', default='cifar10', help=['cifar10','cifar100','tin'])
    # for long-tailed
    # wt decay SGD - 5e - 4
    parser.add_argument('--lt', action='store_true', help='long-tailed or not')
    parser.add_argument('--ratio', default=0.01, type=float, help='Imbalance ratio -- only if imbal=True')

    # args parse
    args = parser.parse_args()
    feature_dim, tmin, tmax, k = args.feature_dim, args.tmin, args.tmax, args.k
    func, knn_t, batch_size, epochs = args.func, args.knn_t, args.batch_size, args.epochs
    exp_params, opti, lr = args.exp_params, args.opti, args.lr
    global_step = 0

    # data prepare
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

    elif args.dataset == 'tin' and args.lt == False:
        args.ratio = 1.0
        train_data = utils.TinyImageNetPair(root=args.datapath+'/tiny-imagenet-200/train/', transform=utils.tin_train_transform)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,drop_last=True)
        memory_data = utils.TinyImageNetPair(root=args.datapath+'/tiny-imagenet-200/train/', transform=utils.test_transform)
        memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_data = utils.TinyImageNetPair(root=args.datapath+'/tiny-imagenet-200/val/', transform=utils.test_transform)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # model setup and optimizer config
    model = Model2(feature_dim).cuda()
    flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
    flops, params = clever_format([flops, params])
    print('# Model Params: {} FLOPs: {}'.format(params, flops))
    optimizer = eval(f'optim.{opti}(model.parameters(), lr={lr}, weight_decay=5e-4)')
    if opti == 'SGD':
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, verbose = True)
        # scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=0, max_epochs=epochs) # for SGD # multistep 160 190
        # scheduler = MultiStepLR(optimizer, milestones=[160, 190], gamma=0.1, verbose= True)
    c = len(train_data.classes)

    # training loop
    data_type = 'BAL' if not args.lt else 'LT'
    results = {'learning_rate': [], 'train_loss': [], 'test_acc@1': [], 'test_acc@5': []}
    save_name_pre = '{}-{}-{}-{}-{}-{}-{}-{}NN-{}-{}'.format(args.dataset,data_type, opti, lr, func, tmin, tmax, k, batch_size, epochs).replace('.','p')

    today_datetime = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S").replace('/','').replace(' ','').replace(':','')
    
    if not os.path.exists('results'):
        os.mkdir('results')
    best_acc = 0.0
    precomp_mask = get_mask_negative(batch_size)
    for epoch in range(1, epochs + 1):
        curr_lr = optimizer.state_dict()['param_groups'][0]['lr']
        train_loss = train(model, train_loader, optimizer, batch_size, tmin, tmax)
        if opti == 'SGD':
            scheduler.step() # for SGD
        results['learning_rate'].append(curr_lr)
        results['train_loss'].append(train_loss)
        test_acc_1, test_acc_5 = test(model, memory_loader, test_loader, knn_t)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        # save statistics
        
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('./results/'+save_name_pre+'_stats_'+today_datetime+'.csv', index_label='epoch')
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            torch.save(model.state_dict(), './results/'+save_name_pre+'_model_'+today_datetime+'.pth')

        global_step += 1
