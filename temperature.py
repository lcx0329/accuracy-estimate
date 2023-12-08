from pyrsistent import get_in
from requests import get
import torch
from torchmetrics import Metric

import torchvision
import torchvision.transforms as transforms
from data_helper import get_data, get_data_c    
from calibration import *


import time
import argparse
from dataset_c import get_cifar_c, get_tiny_c
import numpy as np
from conf import settings
from dataset import TinyImagenet
from model_helper import get_net
from models import *
from toolkit.commons import *
# from toolkit.evaluate import Uncertainty
# from toolkit.evaluate import Metric
import csv
import os
from torch.utils.data import TensorDataset
import torch
from torch.utils.data import DataLoader

def find_ATC_threshold(scores, labels): 
    sorted_idx = np.argsort(scores)
    
    sorted_scores = scores[sorted_idx]
    sorted_labels = labels[sorted_idx]
    
    fp = np.sum(labels==0)
    fn = 0.0
    
    min_fp_fn = np.abs(fp - fn)
    thres = 0.0
    for i in range(len(labels)): 
        if sorted_labels[i] == 0: 
            fp -= 1
        else: 
            fn += 1
        
        if np.abs(fp - fn) < min_fp_fn: 
            min_fp_fn = np.abs(fp - fn)
            thres = sorted_scores[i]
    
    return min_fp_fn, thres


def get_index(testloader, net, idx):
    images = []
    labels = [] 
    for X, y in testloader:
        images.append(X)
        labels.append(y)
    images = torch.cat(images).cuda()
    labels = torch.cat(labels).cuda()
    probs = []
    net.eval()
    with torch.no_grad():
        for X, y in testloader:
            X= X.cuda()
            output = net(X)   
            probs.append(output)

    probs = torch.vstack(probs)
    probs = torch.softmax(probs / args.temperature, dim=1)
    probs = probs.detach().cpu().numpy()
    if (idx == 'entropy'):
        entropy = -Uncertainty.entropy(probs, norm=True)
        return entropy, images, labels
    elif (idx == 'margin'):
        margin = -Uncertainty.margin(probs)
        return margin, images, labels
    elif (idx == 'confidence'):
        confidence = -Uncertainty.confidence(probs)
        return confidence, images, labels
    elif (idx == 'gini'):
        gini = -Uncertainty.gini(probs, norm=True)
        return gini, images, labels
    

def get_basemap(bucket, entropy, images, labels):
    accs = []
    mode_his = []
    consistant_list = []
    for i in range(bucket):
        mode_his.append(i)
        start = -1 / bucket * (bucket - i)
        end = -1 / bucket * (bucket - 1 - i)
        idx = torch.where((entropy >= start) & (entropy < end))[0]
        consistant_list.append(len(idx))
        if (len(idx) == 0):
            accs.append(0)
        else:
            data = images[idx]
            label = labels[idx]
            # if (len(idx) > 50):
            tensorDataset = torch.utils.data.TensorDataset(data, label)
            train_loader = DataLoader(dataset=tensorDataset, batch_size=64, shuffle=False)
            acc = ModelMetric(train_loader).accuracy(net)       
            accs.append(acc)
            # else:
            #     output = net(data)
            #     output = torch.softmax(output, dim=1)
            #     acc = (torch.count_nonzero(torch.argmax(output, dim=1) == label) / len(idx)).cpu().item()
            #     accs.append(acc)
    mode_his = np.asarray(mode_his)
    consistant_list = np.asarray(consistant_list)
    accs = np.asarray(accs)
    basemap = [mode_his, consistant_list, accs]
    return basemap
    
if __name__ == '__main__':
    start = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument('-serverity', type=int, default=1, help='serverity of corruption')
    parser.add_argument('-net', type=str, default="resnet101", help='net')
    parser.add_argument('-data', type=str, default="ImageNet-200", help='CIFAR10 or CIFAR100 or ImageNet-200')
    parser.add_argument('-c', type=str, default="contrast", help='')
    parser.add_argument('-norm', type=int, default=0, help='whether to normalize the data')
    parser.add_argument('-temperature', type=int, default=1, help='temperature of softmax')
    parser.add_argument('-save_path', type=str, default="./result/temp.csv", help='save path')
    parser.add_argument('-bucket', type=int, default=50, help='total bucket')
    parser.add_argument('-num_classes', type=int, default=10)
    parser.add_argument('-index', type=str, default="entropy")
    args = parser.parse_args()
    
    
    testloader = get_data(args.data, 64)
    test_loaders = get_data_c(args.data, 64, args.serverity)
    net = get_net(args.net, args.data)
    
    
    f = open(args.save_path, 'a', encoding='utf-8', newline="")
    csv_writer = csv.writer(f)
    
    bucket = args.bucket
    # 计算测试集的entropy等指标并进行分桶
    index_ori, images_ori, labels_ori = get_index(testloader, net, args.index)
    print(index_ori)
    
    # exit(0)
    metric = ModelMetric(testloader)
    base_acc = metric.accuracy(net)
    pred_probs = metric.get_probs(net, temperature=1).cpu().numpy()
    true_labels = wrapper.loader_to_tensor(testloader)[1].cpu().numpy()
    
    
    calibrator = TempScaling()
    calibrator.fit(pred_probs, true_labels)
    pred_probs = calibrator.calibrate(pred_probs)
    
    pred_probs = torch.from_numpy(pred_probs).cuda()
    true_labels = torch.from_numpy(true_labels)
    entropy_ori = -Uncertainty.confidence(torch.softmax(pred_probs, dim=1), norm=True)
    
    
    basemap_ori = get_basemap(bucket, entropy_ori, images_ori, labels_ori)
    print(basemap_ori[1].sum())
    pred_labels = torch.argmax(pred_probs, dim=-1).cpu()
    train_y = ((pred_labels == true_labels) * 1)
    if ((args.index == "confidence") | (args.index == "margin")):
        _, thres_atc = find_ATC_threshold(-entropy_ori, train_y.cpu().numpy())
    else:
        _, thres_atc = find_ATC_threshold(entropy_ori, train_y.cpu().numpy())
    print(thres_atc)
    print(base_acc)
    csv_writer.writerow(["-----------------------------------------"])
    csv_writer.writerow(["bucket", args.bucket, "index", args.index, "data", args.data, "net", args.net, "base_acc", base_acc, "thres_atc", thres_atc])
    # 计算ood数据集的entropy等指标并进行分桶
    data_types_cifar = ['jpeg_compression',
            'shot_noise',
            'elastic_transform',
            'glass_blur',
            'zoom_blur',
            'impulse_noise',
            'speckle_noise',
            'pixelate',
            'motion_blur',
            'gaussian_blur',
            'frost',
            'defocus_blur',
            'fog',
            'snow',
            'brightness',
            'saturate',
            'gaussian_noise',
            'contrast',
            'spatter']
    data_types_tiny = ['jpeg_compression',
            'shot_noise',
            'elastic_transform',
            'glass_blur',
            'zoom_blur',
            'impulse_noise',
            'pixelate',
            'motion_blur',
            'frost',
            'defocus_blur',
            'fog',
            'snow',
            'brightness',
            'gaussian_noise',
            'contrast']
    if ((args.data == "CIFAR10") | (args.data == "CIFAR100")):
        for data_type in data_types_cifar:
            print(data_type)
            index_ood, images_ood, labels_ood = get_index(test_loaders[data_type], net, args.index)
            
            metric = ModelMetric(test_loaders[data_type])
            pred_probs = metric.get_probs(net, temperature=1).cpu().numpy()
            pred_probs = calibrator.calibrate(pred_probs)
            pred_probs = torch.from_numpy(pred_probs).cuda()
            entropy_ood = -Uncertainty.confidence(torch.softmax(pred_probs, dim=1), norm=True)
            
            
            basemap_ood = get_basemap(bucket, entropy_ood, images_ood, labels_ood)
            if ((args.index == "confidence") | (args.index == "margin")):
                pred_acc_atc = np.count_nonzero(-entropy_ood >= thres_atc) / len(entropy_ood)
            else:
                pred_acc_atc = np.count_nonzero(entropy_ood >= thres_atc) / len(entropy_ood)
            print(basemap_ood[1])
            print(basemap_ori[1])
            # 计算ood数据集的acc 
            acc1 = 0   
            for i in range(args.bucket):
                acc1 += basemap_ood[1][i] * basemap_ori[2][i]
            print(acc1)
            acc1 /= len(images_ood)
            if ((args.index == "entropy") | (args.index == "gini")):
                i = 1;
                while (basemap_ood[1][-i:].sum() == 0):
                    i = i + 1
                print(i)
                acc2 = (basemap_ood[1][-i:].sum() / basemap_ori[1][-i:].sum()) * base_acc
            else:
                i = 1;
                while (basemap_ood[1][:i].sum() == 0):
                    i = i + 1
                print(i)
                acc2 = (basemap_ood[1][:i].sum() / basemap_ori[1][:i].sum()) * base_acc
            print(acc1, acc2, (acc1 + acc2) / 2)
            print(pred_acc_atc)
            true_acc = metric.accuracy(net);
            print(metric.accuracy(net))
            csv_writer.writerow([data_type, abs(true_acc - (acc1 + acc2) / 2) * 100, abs(pred_acc_atc - true_acc) * 100, metric.accuracy(net) * 100])
    else:
        for data_type in data_types_cifar:
            print(data_type)
            if data_type in data_types_tiny:
                index_ood, images_ood, labels_ood = get_index(test_loaders[data_type], net, args.index)
                metric = ModelMetric(test_loaders[data_type])
                pred_probs = metric.get_probs(net, temperature=1).cpu().numpy()
                pred_probs = calibrator.calibrate(pred_probs)
                pred_probs = torch.from_numpy(pred_probs).cuda()
                entropy_ood = -Uncertainty.confidence(torch.softmax(pred_probs, dim=1), norm=True)
                
                basemap_ood = get_basemap(bucket, entropy_ood, images_ood, entropy_ood)
                if ((args.index == "confidence") | (args.index == "margin")):
                    pred_acc_atc = np.count_nonzero(-entropy_ood >= thres_atc) / len(entropy_ood)
                else:
                    pred_acc_atc = np.count_nonzero(entropy_ood >= thres_atc) / len(entropy_ood)
                print(basemap_ori[1])
                print(basemap_ood[1])
                acc1 = 0
                for i in range(bucket):
                    acc1 += basemap_ood[1][i] * basemap_ori[2][i]
                acc1 /= len(images_ood)
                if ((args.index == "entropy") | (args.index == "gini")):
                    i = 1;
                    while (basemap_ood[1][-i:].sum() == 0):
                        i = i + 1
                    print(i)
                    acc2 = (basemap_ood[1][-i:].sum() / basemap_ori[1][-i:].sum()) * base_acc
                else:
                    i = 1;
                    while (basemap_ood[1][:i].sum() == 0):
                        i = i + 1
                    print(i)
                    acc2 = (basemap_ood[1][:i].sum() / basemap_ori[1][:i].sum()) * base_acc
                print(acc1, acc2, (acc1 + acc2) / 2)
                print(pred_acc_atc)
                print(metric.accuracy(net))
                true_acc = metric.accuracy(net);
                csv_writer.writerow([data_type, abs(acc1 - true_acc) * 100, abs(acc2 - true_acc) * 100, abs(true_acc - (acc1 + acc2) / 2) * 100, abs(pred_acc_atc - true_acc) * 100, metric.accuracy(net) * 100])
            else:
                csv_writer.writerow([data_type, "-", "-", "-", "-", "-"])
    end = time.perf_counter()
    elapsed = end - start
    print(f"程序运行时间为{elapsed}毫秒")
    