import logging
import torch
import numpy as np
import random
import argparse
import os

from utils import set_for_logger
from pathlib import Path
from nets import AlexNet_FedGLCL
import copy

import torchvision.transforms as transforms
import pickle as pkl
from utils import OfficeDataset

def test(model, data_loader, loss_fun, device, text_embed):
    model.eval()
    loss_all = 0
    total = 0
    correct = 0

    t = torch.ones([]) * np.log(1 / 0.07)
    t = t.exp()

    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            
            img_embed = model(data)
            img_embed = img_embed / img_embed.norm(dim=-1, keepdim=True)
            similarity = t * img_embed @ text_embed.T
            loss = loss_fun(similarity, target)

            similarity = (100 * img_embed @ text_embed.T).softmax(dim=-1)

            _, pred = torch.max(similarity, 1)            

            loss_all += loss.item()
            total += target.size(0)

            correct += pred.eq(target.view(-1)).sum().item()

        return loss_all / len(data_loader), correct/total

def train_epoch(net_id, net, train_dataloader, epochs, lr, optimizer, weight_decay, device, text_embed):
    logging.info('client training %s' % str(net_id))
    net.train()
    if optimizer == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=weight_decay)
    elif optimizer == 'amsgrad':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=weight_decay,
                               amsgrad=True)
    elif optimizer == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    t = torch.ones([]) * np.log(1 / 0.07)
    t = t.exp()

    for epoch in range(1, epochs+1):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)
            target = target.long()

            img_embed = net(x)

            img_embed = img_embed / img_embed.norm(dim=-1, keepdim=True)
            similarity = t * img_embed @ text_embed.T

            loss = criterion(similarity, target)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / (len(epoch_loss_collector)+ 1e-14)
        logging.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))
    logging.info(' ** Training complete **')
    return epoch_loss

def prepare_data(args):
    data_base_path = './data/'
    transform_office = transforms.Compose([
            transforms.Resize([256, 256]),            
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30,30)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
            transforms.Resize([256, 256]),          
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # amazon
    amazon_trainset = OfficeDataset(data_base_path, 'amazon', transform=transform_office)
    amazon_testset = OfficeDataset(data_base_path, 'amazon', transform=transform_test, train=False)
    # caltech
    caltech_trainset = OfficeDataset(data_base_path, 'caltech', transform=transform_office)
    caltech_testset = OfficeDataset(data_base_path, 'caltech', transform=transform_test, train=False)
    # dslr
    dslr_trainset = OfficeDataset(data_base_path, 'dslr', transform=transform_office)
    dslr_testset = OfficeDataset(data_base_path, 'dslr', transform=transform_test, train=False)
    # webcam
    webcam_trainset = OfficeDataset(data_base_path, 'webcam', transform=transform_office)
    webcam_testset = OfficeDataset(data_base_path, 'webcam', transform=transform_test, train=False)


    amazon_train_loader = torch.utils.data.DataLoader(amazon_trainset, batch_size=args.batch, shuffle=True)
    amazon_test_loader = torch.utils.data.DataLoader(amazon_testset, batch_size=args.batch, shuffle=False)

    caltech_train_loader = torch.utils.data.DataLoader(caltech_trainset, batch_size=args.batch, shuffle=True)
    caltech_test_loader = torch.utils.data.DataLoader(caltech_testset, batch_size=args.batch, shuffle=False)

    dslr_train_loader = torch.utils.data.DataLoader(dslr_trainset, batch_size=args.batch, shuffle=True)
    dslr_test_loader = torch.utils.data.DataLoader(dslr_testset, batch_size=args.batch, shuffle=False)

    webcam_train_loader = torch.utils.data.DataLoader(webcam_trainset, batch_size=args.batch, shuffle=True)
    webcam_test_loader = torch.utils.data.DataLoader(webcam_testset, batch_size=args.batch, shuffle=False)
    
    train_loaders = [amazon_train_loader, caltech_train_loader, dslr_train_loader, webcam_train_loader]
    test_loaders = [amazon_test_loader, caltech_test_loader, dslr_test_loader, webcam_test_loader]


    return train_loaders, test_loaders, [1/4, 1/4, 1/4, 1/4]


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0, help="Random seed")
    parser.add_argument('--partition', type=str, default='noniid', help='the data partitioning strategy')
    parser.add_argument('--comm_round', type=int, default=300, help='number of maximum communication round')
    parser.add_argument('--epochs', type=int, default=5, help='number of local epochs')
    parser.add_argument('--device', type=str, default='cuda:3', help='The device to run the program')

    parser.add_argument('--log_dir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--save_dir', type=str, required=False, default="./weights/", help='Log directory path')

    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--batch', type=int, default=32, help='input batch size for training (default: 64)')
    parser.add_argument('--fl_method', type=str, default='fedglcl')

    args = parser.parse_args()
    return args

def main(args):
    set_for_logger(args)
    logging.info(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device(args.device)

    train_loaders, test_loaders, fed_avg_freqs = prepare_data(args)
    global_model = AlexNet_FedGLCL().to(device)

    from transformers import AutoTokenizer, AlignModel
    classes_strs = ['backpack', 'bike', 'calculator', 'headphones', 'keyboard', 'laptop computer', 'monitor', 'mouse', 'mug', 'projector']
    model = AlignModel.from_pretrained("xxx")  #download checkpoint and change the path 
    tokenizer = AutoTokenizer.from_pretrained("xxx")
    inputs = tokenizer([f"a photo of a {c}" for c in classes_strs], padding=True, return_tensors="pt")
    text_features = model.get_text_features(**inputs)
    text_embed = text_features.detach()
    text_embed = text_embed.to(device)
    text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)

    datasets = ['Amazon', 'Caltech', 'DSLR', 'Webcam']
    client_num = 4

    best_accuarcy = 0
    best_round = 0

    local_models = [copy.deepcopy(global_model).to(device) for idx in range(client_num)]

    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    weight_save_dir = os.path.join(args.save_dir, args.partition, args.fl_method, str(os.getpid()))
    Path(weight_save_dir).mkdir(parents=True, exist_ok=True)
    logging.info('checkpoint will be saved at {}'.format(weight_save_dir))

    for round in range(1, args.comm_round):

        logging.info('----Communication Round: %d -----' % round)
        global_w = global_model.state_dict()

        for i in range(client_num):
            train_epoch(i, local_models[i], train_loaders[i], args.epochs, args.lr, args.optimizer, args.weight_decay, device, text_embed)

        for id in range(client_num):
            model_param = local_models[id].state_dict()
            for key in model_param:
                if id == 0:
                    global_w[key] = model_param[key] * fed_avg_freqs[id]
                else:
                    global_w[key] += model_param[key] * fed_avg_freqs[id]

        global_model.load_state_dict(global_w)
        for i in range(client_num):
            local_models[i].load_state_dict(global_w)

        avg_acc = 0

        logging.info('------------------Testing-----------------')
        for i in range(client_num):
            _, test_acc = test(local_models[i], test_loaders[i], loss_fn, device, text_embed)
            logging.info('>> %s Test accuracy: %f' % (datasets[i], test_acc))
            avg_acc += test_acc
        
        avg_acc /= client_num
        logging.info('>> Average Test accuracy: %f' %avg_acc)

        if avg_acc > best_accuarcy:
            best_accuarcy = avg_acc
            best_round = round

        weight_save_path = os.path.join(weight_save_dir, 'checkpoint_{}.pth'.format(round))
        torch.save(global_model.state_dict(), weight_save_path)


    logging.info(' %d epoch get the best acc %f' % (best_round, best_accuarcy))

if __name__ == '__main__':
    args = get_args()
    main(args)
