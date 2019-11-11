import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# GCN

class GCN(nn.Module):
    def __init__(self, in_dim, hd_dim, out_dim):
        super(GCN, self).__init__()
        self.li0, self.li1 = nn.Linear(in_dim, hd_dim), nn.Linear(hd_dim, out_dim)
    
    def forward(self, adj, x):
        x = F.dropout(x, p=0.5, training = self.training)
        x = self.li0(x)
        x = torch.spmm(adj, x)
        x = F.relu(x)
        x = self.li1(x)
        x = torch.spmm(adj, x)
        return x

# GFNN

class GFNN(nn.Module):
    def __init__(self, in_dim, hd_dim, out_dim):
        super(GFNN, self).__init__()
        self.li0, self.li1 = nn.Linear(in_dim, hd_dim), nn.Linear(hd_dim, out_dim)
        
    def forward(self, adj, x):
        x = F.dropout(x, p=0.5, training = self.training)
        x = self.li0(x)
        x = torch.spmm(adj, torch.spmm(adj, x))
        x = self.li1(F.relu(x))
        return x

# MLP

class MLP(nn.Module):
    def __init__(self, in_dim, hd_dim=None, out_dim=None):
        super(MLP, self).__init__()
        self.hd_dim = hd_dim
        if hd_dim is not None:
            self.li0, self.li1 = nn.Linear(in_dim, hd_dim), nn.Linear(hd_dim, out_dim)
        else:
            self.li = nn.Linear(in_dim, out_dim)
        
    def forward(self, x):
        x = F.dropout(x, p=0.5, training=self.training)
        if self.hd_dim is not None:
            x = self.li1(F.relu(self.li0(x)))
        else:
            x = self.li(x)
        return x

# GraphRNA_RW

class pro_lstm_featwalk(nn.Module):
    def __init__(self, nfeat, nhid, nclass, num_paths, path_length, dropout, multilabel=False):
        super(pro_lstm_featwalk, self).__init__()
        target_size = nclass
        self.path_length = path_length
        self.num_paths = num_paths
        self.dropout = nn.Dropout(p=dropout)
        self.droprate = dropout
        self.multilabel = multilabel
        self.nonlinear = nn.ReLU()
        self.first = nn.Sequential(
            self.dropout,
            nn.Linear(nfeat, nhid * 4),
            #nn.BatchNorm1d(nhid * 4),
            self.nonlinear
        )
        '''
        self.mode = 'LSTM'
        if self.mode == 'LSTM':
            self.lstm = nn.LSTM(input_size=nhid * 4,
                               hidden_size=nhid,
                               num_layers=1,
                               #dropout=dropout,
                               bidirectional=True)
        elif self.mode == 'GRU':
            self.gru = nn.GRU(input_size=nhid * 4,
                              hidden_size=nhid,
                              num_layers=1,
                              #dropout=dropout,
                              bidirectional=True)
        '''
        self.hidden2tag = nn.Sequential(
            #nn.BatchNorm1d(nhid * 8),
            #self.nonlinear,
            #self.dropout,
            nn.Linear(nhid * 8, target_size)
        )
        #self.out = nn.Hardtanh()  # Sigmoid, Hardtanh

    def forward(self, x):  # idxinv
        ####print('forward begin', x.size())
        x = self.first(x)
        x = x.view(-1, self.path_length, x.size(1)).transpose_(0, 1)
        ####print('before lstm', x.size())
        #_, batch_size, hiddenlen = x.size()
        #selfloop1 = torch.mean(x.view(self.path_length, int(batch_size / self.num_paths), self.num_paths, hiddenlen), dim=2)[0]
        '''
        if self.mode == 'LSTM':
            outputs, (ht, ct) = self.lstm(x)
        elif self.mode == 'GRU':
            outputs, ht = self.gru(x)
        '''
        ####print('after lstm', outputs.size())
        outputs = x
        _, batch_size, hidlen = outputs.size()
        outputs = outputs.view(self.path_length, int(batch_size / self.num_paths), self.num_paths, hidlen).mean(dim=2)
        ####print('after view', outputs.size())
        selfloop = outputs[0]
        #selfloop1 = x[0].view(int(batch_size / self.num_paths), self.num_paths, x.size(2)).mean(dim=1)

        #weight = torch.from_numpy(np.power(self.beta, range(self.path_length))).float().cuda()
        #weight[0] = self.path_length - 1
        #outputs = torch.mean(torch.matmul(outputs.transpose_(0, 1).transpose_(1, 2), torch.diag(weight)), dim=2)
        #outputs = torch.cat((outputs, selfloop), dim=1)

        outputs = torch.cat((outputs.mean(dim=0), selfloop), dim=1)  #
        ####print('before hidden2tag', outputs.size())
        outputs = self.hidden2tag(outputs)
        ####print(outputs.size())
        ####exit()
        return outputs
        #outputs = outputs.transpose_(1, 2).contiguous().view(senlen * num_paths, batch_size / num_paths, lablelen)
        #return torch.mean(torch.mean(outputs, dim=0),  dim=1)
    def embedding(self, x):
        x = self.first(x)
        x = x.view(-1, self.path_length, x.size(1)).transpose_(0, 1)
        '''
        if self.mode == 'LSTM':
            outputs, (ht, ct) = self.lstm(x)
        elif self.mode == 'GRU':
            outputs, ht = self.gru(x)
        '''
        outputs = x
        _, batch_size, hidlen = outputs.size()
        outputs = outputs.view(self.path_length, int(batch_size / self.num_paths), self.num_paths, hidlen).mean(dim=2)
        selfloop = outputs[0]
        outputs = torch.cat((outputs.mean(dim=0), selfloop), dim=1)

        return outputs


# GraphRNA_RW_FULL

class pro_lstm_featwalk_full(nn.Module):
    def __init__(self, nfeat, nhid, nclass, num_paths, path_length, dropout, multilabel=False):
        super(pro_lstm_featwalk_full, self).__init__()
        target_size = nclass
        self.path_length = path_length
        self.num_paths = num_paths
        self.dropout = nn.Dropout(p=dropout)
        self.droprate = dropout
        self.multilabel = multilabel
        self.nonlinear = nn.Tanh()
        self.first = nn.Sequential(
            nn.Linear(nfeat, nhid * 4),
            nn.BatchNorm1d(nhid * 4),
            self.nonlinear,
            self.dropout,
        )
        self.mode = 'LSTM'
        if self.mode == 'LSTM':
            self.lstm = nn.LSTM(input_size=nhid * 4,
                               hidden_size=nhid,
                               num_layers=1,
                               #dropout=dropout,
                               bidirectional=True)
        elif self.mode == 'GRU':
            self.gru = nn.GRU(input_size=nhid * 4,
                              hidden_size=nhid,
                              num_layers=1,
                              #dropout=dropout,
                              bidirectional=True)

        self.hidden2tag = nn.Sequential(
            nn.BatchNorm1d(nhid * 4),
            self.nonlinear,
            self.dropout,
            nn.Linear(nhid * 4, target_size)
        )
        #self.out = nn.Hardtanh()  # Sigmoid, Hardtanh

    def forward(self, x):  # idxinv
        x = self.first(x)
        x = x.view(-1, self.path_length, x.size(1)).transpose_(0, 1)
        #_, batch_size, hiddenlen = x.size()
        #selfloop1 = torch.mean(x.view(self.path_length, int(batch_size / self.num_paths), self.num_paths, hiddenlen), dim=2)[0]

        if self.mode == 'LSTM':
            outputs, (ht, ct) = self.lstm(x)
        elif self.mode == 'GRU':
            outputs, ht = self.gru(x)
        _, batch_size, hidlen = outputs.size()
        outputs = outputs.view(self.path_length, int(batch_size / self.num_paths), self.num_paths, hidlen).mean(dim=2)
        selfloop = outputs[0]
        #selfloop1 = x[0].view(int(batch_size / self.num_paths), self.num_paths, x.size(2)).mean(dim=1)

        #weight = torch.from_numpy(np.power(self.beta, range(self.path_length))).float().cuda()
        #weight[0] = self.path_length - 1
        #outputs = torch.mean(torch.matmul(outputs.transpose_(0, 1).transpose_(1, 2), torch.diag(weight)), dim=2)
        #outputs = torch.cat((outputs, selfloop), dim=1)

        outputs = torch.cat((outputs.mean(dim=0), selfloop), dim=1)  #
        outputs = self.hidden2tag(outputs)
        return outputs
        #outputs = outputs.transpose_(1, 2).contiguous().view(senlen * num_paths, batch_size / num_paths, lablelen)
        #return torch.mean(torch.mean(outputs, dim=0),  dim=1)
    def embedding(self, x):
        x = self.first(x)
        x = x.view(-1, self.path_length, x.size(1)).transpose_(0, 1)
        if self.mode == 'LSTM':
            outputs, (ht, ct) = self.lstm(x)
        elif self.mode == 'GRU':
            outputs, ht = self.gru(x)
        _, batch_size, hidlen = outputs.size()
        outputs = outputs.view(self.path_length, int(batch_size / self.num_paths), self.num_paths, hidlen).mean(dim=2)
        selfloop = outputs[0]
        outputs = torch.cat((outputs.mean(dim=0), selfloop), dim=1)

        return outputs