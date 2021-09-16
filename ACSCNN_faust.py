# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 16:51:25 2019

@author: Michael
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Parameter

class ACSConv(nn.Module):
    def __init__(self,in_size,out_size,n_angles=8, K=15,bias=True):
        super().__init__()
        self.in_size=in_size
        self.out_size=out_size
        self.n_angles=n_angles
        self.K=K
        self.weight=Parameter(torch.Tensor(K,n_angles*in_size,out_size))
    
        nn.init.xavier_uniform_(self.weight)
        
        if bias:
            self.bias=Parameter(torch.Tensor(out_size))
            nn.init.constant_(self.bias,0)
        else:
            self.register_parameter('bias',None)
        
    def forward(self,x,Ls):
        N=x.size(0)
        Tx_0=x.repeat(self.n_angles,1)
        out=torch.matmul(Tx_0.view(self.n_angles,N,self.in_size).permute(1,0,2).contiguous().view(N,self.n_angles*self.in_size),self.weight[0])
        
        if self.K > 1:
            Tx_1=torch.matmul(Ls,Tx_0) #[H*N,P]
            out=out+torch.matmul(Tx_1.view(self.n_angles,N,self.in_size).permute(1,0,2).contiguous().view(N,self.n_angles*self.in_size),self.weight[1])
            
        for k in range(2,self.K):
            Tx_2=torch.matmul(2*Ls,Tx_1)-Tx_0
            out=out+torch.matmul(Tx_2.view(self.n_angles,N,self.in_size).permute(1,0,2).contiguous().view(N,self.n_angles*self.in_size),self.weight[k])
            Tx_0,Tx_1=Tx_1,Tx_2
        
        if self.bias is not None:
            out=out+self.bias
        
        return out

# splinecnn arc
class ACSCNN(nn.Module):
    def __init__(self,n_desc,n_class):
        super().__init__()
        self.n_desc=n_desc
        self.n_class=n_class
        
        self.conv1=ACSConv(n_desc,64)
        self.bn1=nn.BatchNorm1d(64)
        self.conv2=ACSConv(64,64)
        self.bn2=nn.BatchNorm1d(64)
        self.conv3=ACSConv(64,64)
        self.bn3=nn.BatchNorm1d(64)
        self.conv4=ACSConv(64,64)
        self.bn4=nn.BatchNorm1d(64)
        self.conv5=ACSConv(64,64)
        self.bn5=nn.BatchNorm1d(64)
        self.conv6=ACSConv(64,64)
        self.bn6=nn.BatchNorm1d(64)
        self.fc2=nn.Linear(64,256)
        self.fc3=nn.Linear(256,n_class)
        
    def forward(self,x,L):
        x=F.relu(self.bn1(self.conv1(x,L)))
        x=F.relu(self.bn2(self.conv2(x,L)))
        x=F.relu(self.bn3(self.conv3(x,L)))
        x=F.relu(self.bn4(self.conv4(x,L)))
        x=F.relu(self.bn5(self.conv5(x,L)))
        x=F.relu(self.bn6(self.conv6(x,L)))
        x=F.relu(self.fc2(x))
        x=F.dropout(x,training=self.training)
        
        return self.fc3(x)

# ACNN arc
#class ACSCNN(nn.Module):
#   def __init__(self,n_desc,n_class):
#       super().__init__()
#       self.n_desc=n_desc
#       self.n_class=n_class
#       
#       self.fc1=nn.Linear(n_desc,32)
#       self.bn0=nn.BatchNorm1d(32)
#       self.conv1=ACSConv(32,64)
#       self.bn1=nn.BatchNorm1d(64)
#       self.conv2=ACSConv(64,64)
#       self.bn2=nn.BatchNorm1d(64)
#       self.conv3=ACSConv(64,128)
#       self.bn3=nn.BatchNorm1d(128)
#       self.fc2=nn.Linear(128,256)
#       self.fc3=nn.Linear(256,n_class)
#       
#   def forward(self,x,L):
#       x=F.relu(self.bn0(self.fc1(x)))
#       x=F.relu(self.bn1(self.conv1(x,L)))
#       x=F.relu(self.bn2(self.conv2(x,L)))
#       x=F.relu(self.bn3(self.conv3(x,L)))
#       x=F.relu(self.fc2(x))
#       x=F.dropout(x,training=self.training)
#       
#       return self.fc3(x)  

## 数据加载
import os
import time
import h5py
import numpy as np
import scipy.sparse as sp
from torch.utils.data import Dataset

# import matlab files in python
def load_matlab_file(path_file, name_field):
    """
    load '.mat' files
    inputs:
        path_file, string containing the file path
        name_field, string containing the field name (default='shape')
    warning:
        '.mat' files should be saved in the '-v7.3' format
    """
    db = h5py.File(path_file, 'r')
    ds = db[name_field]
    try:
        if 'ir' in ds.keys():
            data = np.asarray(ds['data'])
            ir = np.asarray(ds['ir'])
            jc = np.asarray(ds['jc'])
            out = sp.csc_matrix((data, ir, jc)).astype(np.float32) #转为csr_matrix，加速矩阵计算
    except AttributeError:
        # Transpose in case is a dense matrix because of the row- vs column- major ordering between python and matlab
        out = np.asarray(ds).astype(np.float32).T
    db.close()
    return out
            

# imports labels from matlab to python and converts them to the python indexing
# (labels start with 0 instead of 1)
def load_labels(fname):
    tmp = load_matlab_file(fname, 'labels')
    tmp -= 1.0
    tmp = tmp.astype(np.int32)
    return tmp.flatten() 


class MyDataset(Dataset):
    def __init__(self, files, path_descs, path_ALBO, path_labels, transform=None):
        # train / test instances
        self.files=files
        # path to (pre-computed) descriptors
        self.path_descs = path_descs
        # path to (pre-computed) ALBO matrix
        self.path_ALBO = path_ALBO
        # path to labels
        self.path_labels = path_labels
        
        self.transform=transform
        
        # loading train / test names
        with open(self.files, 'r') as f:
            self.names = [line.rstrip() for line in f]
        
        # loading the descriptors
        self.descs = []
        print("[i] loading descs... "),
        tic = time.time()
        for name in self.names:
            tmp = load_matlab_file(os.path.join(self.path_descs, name), 'desc')
            self.descs.append(tmp.astype(np.float32))
        print("%02.2fs" % (time.time() - tic))
        
        # loading the ALBO
        self.patch_ops = []
        print("[i] loading ALBO matrix... "),
        tic = time.time()
        for name in self.names:
            M = load_matlab_file(os.path.join(self.path_ALBO, name), 'L')
            self.patch_ops.append(M)
        print("%02.2fs" % (time.time() - tic))
        
        # loading the labels
        self.labels= []
        print("[i] loading labels... "),
        tic = time.time()
        for name in self.names:
            self.labels.append(load_labels(os.path.join(self.path_labels, name)))
        print("%02.2fs" % (time.time() - tic))

        
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self,idx):
       #打乱顺序
        sample={'name':self.names[idx],
                'desc':self.descs[idx],
                'patch_op':self.patch_ops[idx],
                'label':self.labels[idx]}
        
        if self.transform:
            sample=self.transform(sample)
            
        return sample


def spy_sparse2torch_sparse(data):
#    """
#    :param data: a scipy sparse csc matrix
#    :return: a sparse torch tensor
#    """
    samples=data.shape[0]
    features=data.shape[1]
    values=data.data
    coo_data=data.tocoo()
    indices=torch.LongTensor([coo_data.row,coo_data.col])
    t=torch.sparse.FloatTensor(indices,torch.from_numpy(values).float(),[samples,features])
    return t

class ToTensor(object):
#    将numpy数据转为pytorch tensor
#    注意:patch_op为scipy csr_matrix
    def __call__(self,sample):
        return {'name':sample['name'],
                'desc':torch.from_numpy(sample['desc']),
                'patch_op': spy_sparse2torch_sparse(sample['patch_op']),
                'label':torch.from_numpy(sample['label']).long()}
        
#日志文件
import logging
from logging import handlers

class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }#日志级别关系映射

    def __init__(self,filename,level='info',when='D',backCount=3,fmt='%(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)#设置日志格式
        self.logger.setLevel(self.level_relations.get(level))#设置日志级别
        sh = logging.StreamHandler()#往屏幕上输出
        sh.setFormatter(format_str) #设置屏幕上显示的格式
        th = handlers.TimedRotatingFileHandler(filename=filename,when=when,backupCount=backCount,encoding='utf-8')#往文件里写入#指定间隔时间自动生成文件的处理器
        #实例化TimedRotatingFileHandler
        #interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
        # S 秒
        # M 分
        # H 小时、
        # D 天、
        # W 每星期（interval==0时代表星期一）
        # midnight 每天凌晨
        th.setFormatter(format_str)#设置文件里写入的格式
        self.logger.addHandler(sh) #把对象加到logger里
        self.logger.addHandler(th)        


## 实验参数设置
name_exp = 'aniso_norm_laplacian_K=15_const'  # default: exp00
# path to the folder containing the code:
path_code = os.getcwd()
# path to the folder containing the data:
path_data=os.path.join(path_code,'FAUST')
# input paths (where to load data from):
paths = dict()
# path to the input descriptors
paths['descs'] = os.path.join(path_data, 'descs','const')
# path to the ALBO matrix
paths['patch_ops'] = os.path.join(path_data, 'aniso_norm_laplacian')
 
# path to the labels
paths['labels'] = os.path.join(path_data, 'labels')

paths['exp'] = os.path.join(path_data, 'experiments', name_exp)
paths['pred'] = os.path.join(paths['exp'],'preds')
if not os.path.isdir(paths['pred']):
        os.makedirs(paths['pred'])


num_epochs=100

log = Logger(os.path.join(paths['exp'],'info.log'), level='info')
log.logger.info(paths['patch_ops'])

## 开始训练
train_datasets=MyDataset(os.path.join(path_data, 'files_train.txt'),
                   paths['descs'],
                   paths['patch_ops'],
                   paths['labels'],
                   ToTensor())

test_datasets=MyDataset(os.path.join(path_data, 'files_test.txt'),
                   paths['descs'],
                   paths['patch_ops'],
                   paths['labels'],
                   ToTensor())

n_desc=train_datasets[1]['desc'].size(1)
n_class=6890 # for FAUST dataset, reference shape has 6890 vertices


#定义模型
model=ACSCNN(n_desc,n_class)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=model.to(device)
optimizer=optim.Adam(model.parameters())
criterion=torch.nn.CrossEntropyLoss()

import copy

best_acc=0.0
best_epoch=0.0
train_epoch_loss=[]
test_epoch_acc=[]

for epoch in range(num_epochs):
    log.logger.info('Epoch {}/{}'.format(epoch,num_epochs-1))
    log.logger.info('-'*10)
    
    tic=time.time()
    #training
    model.train()
    running_loss=0.0
    running_corrects=0
    epoch_loss=0.0
    epoch_acc=0.0
    n_samples=0.0
    
    idxs=np.arange(len(train_datasets))
    np.random.shuffle(idxs)
    
    for idx in idxs:
        sample=train_datasets[idx]
        y=sample['desc'].to(device)
        L=sample['patch_op'].to(device)
        label=sample['label'].to(device)

        optimizer.zero_grad()
        outputs=model(y,L)
        loss=criterion(outputs,label)
        loss.backward()
        optimizer.step()
        _,preds=torch.max(outputs,1)
        
        n_samples+=label.size(0)
        running_loss+=loss.item()
        running_corrects+=torch.sum(preds==label).item()
        
    epoch_loss=running_loss/len(train_datasets)
    epoch_acc=running_corrects/n_samples
    
    train_epoch_loss.append(epoch_loss)#保存训练每次迭代的误差

    log.logger.info('train Loss :{:.4f} Acc:{:.4f} Time:{:.2f}s'.format(epoch_loss,epoch_acc,time.time()-tic))

    #testing
    tic=time.time()
    model.eval()
    running_loss=0.0
    running_corrects=0
    n_samples=0.0
    
    with torch.no_grad():
        for i,sample in enumerate(test_datasets):
            y=sample['desc'].to(device)
            L=sample['patch_op'].to(device)
            label=sample['label'].to(device)
            outputs=model(y,L)
            _,preds=torch.max(outputs,1)
            loss=criterion(outputs,label)
            
            n_samples+=label.size(0)
            running_loss+=loss.item()
            running_corrects+=torch.sum(preds==label).item()
    
    
    epoch_loss=running_loss/len(test_datasets)
    epoch_acc=running_corrects/n_samples
    
    test_epoch_acc.append(epoch_acc)
    
    log.logger.info('test Loss :{:.4f} Acc:{:.4f} Time:{:.2f}s'.format(epoch_loss,epoch_acc,time.time()-tic))
    

import scipy.io as sio

model.eval()

with torch.no_grad():
    for i, sample in enumerate(test_datasets):
        y=sample['desc'].to(device)
        Ls=sample['patch_op'].to(device)
        label=sample['label'].to(device)

        outputs=model(y,Ls)
        
        _,preds=torch.max(outputs,1)
        sio.savemat(os.path.join(paths['pred'],sample['name']),{'preds':preds.to('cpu').numpy().astype(np.int16)})
        
#保存train_epoch_loss和test_epoch_acc
sio.savemat(os.path.join(paths['pred'],'train_epoch_loss'),{'train_epoch_loss':np.array(train_epoch_loss).astype(np.float32)})
sio.savemat(os.path.join(paths['pred'],'test_epoch_acc'),{'test_epoch_acc':np.array(test_epoch_acc).astype(np.float32)})
