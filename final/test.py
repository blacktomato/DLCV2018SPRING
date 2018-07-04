import os
import scipy.misc
import numpy as np
import argparse
import torch
from torch.utils.data.dataset import Dataset
import torch.utils.data as Data
from PIL import Image
import torchvision.transforms as trans
import torch.nn as nn
import utils
from tqdm import tqdm
from collections import namedtuple, OrderedDict
import time
import quant
import copy
class TestDataset(Dataset):
    def __init__(self, img):
        self.x = torch.from_numpy(img).float()
    def __getitem__(self, index):
        return self.x[index]
    def __len__(self):
        return len(self.x)

def get_data_loader(x, batch_size = 32, shuffle = False):
    dataset = TestDataset(x)
    loader = Data.DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        num_workers = 8
    )
    return loader



def valid(net, criterion, loader):
    pbar = tqdm(iter(loader))
    net.eval()
    correct = 0
    total_loss = 0
    count = 0
    for x_batch, y_batch in pbar:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
 
        pred,_ = net(x_batch)

        loss = criterion(pred, y_batch)
        
        total_loss += loss.item()
        _, pred_class = torch.max(pred, 1)
        correct += (pred_class == y_batch).sum().item()

        count += len(x_batch)

        pbar.set_description('Validation stage: Avg loss: {:.4f}; Avg acc: {:.2f}%'.\
            format(total_loss / count, correct / count * 100))
    acc = correct / count * 100
    return acc

def read_test_data(path):
    #if(os.path.isfile('preproc_data/test.npz')):
    #    dic = np.load('preproc_data/test.npz')
    #    x = dic['x']
    #    return x
    #else:
    img_name_list = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    img_name_list.sort()
    x = []

    for img_name in img_name_list:
        img_path = os.path.join(path, img_name)
        img = scipy.misc.imread(img_path) # 3 x 218 x 178
        x.append(Image.fromarray(img))
    #np.savez('preproc_data/test', x = np.array(x))
    #return np.array(x)
    return x

def test(x_test_all):
    net = torch.load(args.model).to(device)
    net.eval()
    mapping = np.load("./preproc_data/inv_map.npz")
    mapping = mapping['inv_map'].reshape(1)[0] 

    f = open(args.output_path, 'w')
    f.write('id,ans\n')
    transform = trans.Compose([
                trans.CenterCrop(120),
                trans.ToTensor(),
                trans.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
                ])
    for i, x_test in enumerate(x_test_all):
        """
        x_test = torch.from_numpy(x_test).float().to(device) / 255.0
        """
        x_test = transform(x_test)
        x_test = x_test.unsqueeze(0).to(device)

        pred = net(x_test)[0].detach()
        _, pred_class = torch.max(pred, 1)

        #print(i,end='\r')
        f.write('{},{}\n'.format(i + 1, mapping[pred_class.item()]))
    f.close()
def cal_val_acc(net):

    net = net.to(device)
    net.eval()
    mapping = np.load(os.path.join('preproc_data','map.npz'))['map'].reshape(1)[0]
    criterion = nn.CrossEntropyLoss()

    x_val, y_val = utils.read_preproc_data(os.path.join('preproc_data', 'val.npz'))
    val_loader = utils.get_data_loader(x_val, y_val, mapping, data_aug = False, batch_size = 64, shuffle = False)
    val_acc = valid(net, criterion, val_loader)

    return val_acc
def model_size_distribution(net, bn_bits = 32, param_bits = 32):
    def _num_of_param(m_shape):
        return m_shape.numel()
   
   
    
    bn2d = 0
    bn1d = 0
    conv = 0
    linear = 0
    total = 0
    for m in net.modules():
        if(isinstance(m, nn.BatchNorm2d)):
            bn2d += m.weight.numel() * bn_bits        
        if(isinstance(m, nn.BatchNorm1d)):
            bn1d += m.weight.numel() * bn_bits
        if(isinstance(m, nn.Conv2d)):
            conv += m.weight.numel() * param_bits        
        if(isinstance(m, nn.Linear)):
            linear += m.weight.numel() * param_bits
    total = bn2d + bn1d + conv + linear
    print('BatchNorm2d: {} MB'.format(bn2d / 1e6 / 8))
    print('BatchNorm1d: {} MB'.format(bn1d / 1e6 / 8))
    print('Conv2d: {} MB'.format(conv / 1e6 / 8))
    print('Linear: {} MB'.format(linear / 1e6 / 8))
    print('Total: {} MB'.format((total) / 1e6 / 8))
    return total/1e6/8
'''
def test_batch(loader):
    net = torch.load(args.model).to(device)
    net.eval()
    f = open(args.output_path, 'w')
    f.write('id,ans\n')
    counter = 1
    for x_test in loader:
        x_test = x_test.to(device) / 255.0
        pred = net(x_test).detach()
        _, pred_classes = torch.max(pred,1)
        pred_classes = pred_classes.cpu().numpy()

        for pred_class in pred_classes:
            print(counter)
            f.write('{},{}\n'.format(counter, pred_class))
            counter += 1
    f.close
'''

def quantize(model_raw, bn_bits, param_bits, quant_method = 'log'):

    if param_bits < 32:
        state_dict = model_raw.state_dict()
        state_dict_quant = OrderedDict()
        sf_dict = OrderedDict()
        for k, v in state_dict.items():
            #print(len(v.size()))
            if 'running' in k:
                if bn_bits >=32:
                    print("Ignoring {}".format(k))
                    state_dict_quant[k] = v
                    continue
                else:
                    bits = bn_bits
            else:
                bits = param_bits

            if quant_method == 'linear':
                sf = bits - 1. - quant.compute_integral_part(v, overflow_rate=0.0)
                v_quant  = quant.linear_quantize(v, sf, bits=bits)
            elif quant_method == 'log':
                v_quant = quant.log_minmax_quantize(v, bits=bits)
            elif quant_method == 'minmax':
                v_quant = quant.min_max_quantize(v, bits=bits)
            else:
                v_quant = quant.tanh_quantize(v, bits=bits)
            state_dict_quant[k] = v_quant
            #print(k, bits)
            #print(v_quant)
           
        model_raw.load_state_dict(state_dict_quant)
    return model_raw
def quantize_CNN(model_raw, quant_method = 'log'):
    bn_bits = args.bn_bits
    param_bits = args.param_bits

    bn2d = 0
    bn1d = 0
    conv = 0
    linear = 0


    if param_bits < 32:
        state_dict = model_raw.state_dict()
        state_dict_quant = OrderedDict()
        sf_dict = OrderedDict()
        for k, v in state_dict.items():
            bn_bits = args.bn_bits
            param_bits = args.param_bits
            if(len(v.size())==4):   ####################### if CNN
                print('quantize: {}'.format(v.size()))
                if 'running' in k:
                    if bn_bits >=32:
                        #print("Ignoring {}".format(k))
                        state_dict_quant[k] = v
                        continue
                    else:
                        bits = bn_bits
                else:
                    bits = param_bits
    
                if quant_method == 'linear':
                    sf = bits - 1. - quant.compute_integral_part(v, overflow_rate=0.0)
                    v_quant  = quant.linear_quantize(v, sf, bits=bits)
                elif quant_method == 'log':
                    v_quant = quant.log_minmax_quantize(v, bits=bits)
                elif quant_method == 'minmax':
                    v_quant = quant.min_max_quantize(v, bits=bits)
                else:
                    v_quant = quant.tanh_quantize(v, bits=bits)
                state_dict_quant[k] = v_quant

            else:                   ################### if not CNN
                #print('not quantize: {}'.format(v.size()))
                bn_bits = 32
                param_bits = 32
                if 'running' in k:
                    if bn_bits >=32:
                        #print("Ignoring {}".format(k))
                        state_dict_quant[k] = v
                        continue
                    else:
                        bits = bn_bits
                else:
                    bits = param_bits
    
                if quant_method == 'linear':
                    sf = bits - 1. - quant.compute_integral_part(v, overflow_rate=0.0)
                    v_quant  = quant.linear_quantize(v, sf, bits=bits)
                elif quant_method == 'log':
                    v_quant = quant.log_minmax_quantize(v, bits=bits)
                elif quant_method == 'minmax':
                    v_quant = quant.min_max_quantize(v, bits=bits)
                else:
                    v_quant = quant.tanh_quantize(v, bits=bits)
                state_dict_quant[k] = v_quant
                #print(k, bits)
                #print(v_quant)  
           
        model_raw.load_state_dict(state_dict_quant)
    return model_raw
def prune(net):
    count = 0
    for p in list(net.parameters()):
        print(p.data)
        if(p.data < 0.0001):
            count += 1
            p.data = 0
    print(count)
    return net
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Testing and save result')
    parser.add_argument('-i', '--input_dir', default = os.path.join('../../DLCV_FinalProject/dataset', 'test'), help = 'Test image directory')
    parser.add_argument('-m', '--model', help = 'Select which model to test')
    parser.add_argument('-o', '--output_path', default = 'result/result.csv', help = 'Saved file path')
    parser.add_argument('-bn', '--bn_bits', type = int, default = 16, help = 'Quantized number of bits(bn)')
    parser.add_argument('-p', '--param_bits', type = int, default = 16, help = 'Quantized number of bits(param)')
    parser.add_argument('-qm', '--quant_method', type = int, default = 1, help = 'Quantization method')
    parser.add_argument('-d', '--device_id', type = int, default = 0, help = 'Set GPU device')
    args = parser.parse_args()
    
    x_test_all = read_test_data(path = args.input_dir)

    device = torch.device("cuda:{}".format(args.device_id))
    method_list = ['linear', 'log', 'minmax', 'other']
    
    net = torch.load(args.model).to(device)
  
    total = model_size_distribution(net)
    cal_val_acc(net)
    q_net = quantize(copy.deepcopy(net), quant_method = method_list[args.quant_method], bn_bits = 9, param_bits = 9)
    torch.save(q_net, 'saved_model/quantized_depth_fire.pth')
    cal_val_acc(q_net)
    test(x_test_all)

    '''q_net = quantize(copy.deepcopy(net), quant_method = method_list[args.quant_method], bn_bits = 2, param_bits = 2)
    for p in q_net.parameters():
        print(p.size())
        print(p)
    
    def unique(tensor):
        t, idx = np.unique(tensor.numpy(), return_inverse=True)
        return torch.from_numpy(t), torch.from_numpy(idx)
    q, nq = [],[]
    for i, p in enumerate(q_net.parameters()):
        #print(i)
        #print(i,p.size())
        q.append(len(unique(p.data.cpu())[0]))
    for p in net.parameters():
        nq.append(len(unique(p.data.cpu())[0]))
    for i,j in zip(q,nq):
        print(i,j)
    '''
    '''
    f_name = 'fire'
    f = open('q_result/{}'.format(f_name),'w')
    f.close()
    
    for i in range(0,31):
        q = 32 - i
        print('hi')
        net = torch.load(args.model).to(device)
        q_net = quantize(net, quant_method = method_list[args.quant_method], bn_bits = q, param_bits = q)
        #q_net = quantize(copy.deepcopy(net), quant_method = method_list[args.quant_method], bn_bits = q, param_bits = q)
        acc = cal_val_acc(q_net)

        f = open('q_result/{}'.format(f_name),'a')
        print(total, total/32*q)
        f.write('{},{},{}\n'.format(q,acc,total/32*q))
        f.close()
    '''
    

    #torch.save(net.state_dict(), 'dict_model/test.model')

    #model_size_distribution(net)
    #p_net = prune(net)
    #q_net = quantize(copy.deepcopy(net), quant_method = method_list[args.quant_method])
    #cal_val_acc(q_net)
    #for p in q_net.parameters():
    #    if(len(p.size())==4):
    #        print(p)
    #cal_val_acc(q_net)
    #torch.save(q_net, 'half.pth')

    #cal_val_acc(net)
    

    #x_test_all = read_test_data(path = args.input_dir)
    #test(x_test_all)

    '''
        if want to test with batch, comment line 86 and 87,
        and run below two lines
    '''
    #loader = get_data_loader(x_test_all)
    #test_batch(loader)
