import scipy.io as sio
import numpy as np
from scipy.sparse import csc_matrix
import scipy.sparse as sp
from sklearn.preprocessing import normalize
import scipy.sparse.linalg as LA
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import sklearn.metrics as metrics

from model import GCN, GFNN, MLP, pro_lstm_featwalk, pro_lstm_featwalk_full
from random_walk import walk_dic_featwalk
from utils import output_csv, subsample
import time
import sys, os

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='ATTR_RW')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--dataset', type=str, default='BlogCatalog')
parser.add_argument('--proportion', type=float, default=0.25)
parser.add_argument('--saved', action='store_true', default=False)
parser.add_argument('--output_file', type=str, default='toy')
parser.add_argument('--times_features', action='store_true', default=False)
parser.add_argument('--subsample', action='store_true', default=False)
parser.add_argument('--data_path', type=str, default='.')
args = parser.parse_args()

class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def writelines(self, datas):
       self.stream.writelines(datas)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)

print(args)

output_file = 'results/' + args.output_file + '_' + str(args.seed) + '.csv'

np.random.seed(args.seed)
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    device = 'cuda:' + str(args.gpu)
    torch.cuda.manual_seed(args.seed)
else:
    device = 'cpu'

if args.dataset.lower() == 'blogcatalog':
    mat_name = args.data_path + '/data/BlogCatalog.mat'
    nb_classes = 6
elif args.dataset.lower() == 'flickr':
    mat_name = args.data_path + '/data/Flickr.mat' #'data/Flickr_SDM.mat' 
    nb_classes = 9

mat_contents = sio.loadmat(mat_name)

adj = mat_contents['Network']
features = mat_contents['Attributes']

def concatenate_csc_matrices_by_columns(matrix1, matrix2):
    assert matrix1.shape[0] == matrix2.shape[0]
    new_data = np.concatenate((matrix1.data, matrix2.data))
    new_indices = np.concatenate((matrix1.indices, matrix2.indices))
    new_ind_ptr = matrix2.indptr + len(matrix1.data)
    new_ind_ptr = new_ind_ptr[1:]
    new_ind_ptr = np.concatenate((matrix1.indptr, new_ind_ptr))
    
    return csc_matrix((new_data, new_indices, new_ind_ptr))

def concatenate_sparse_matrices_by_rows(matrix1, matrix2):
    assert matrix1.shape[1] == matrix2.shape[1]
    
    return concatenate_csc_matrices_by_columns(matrix1.transpose().tocsc(), matrix2.transpose().tocsc()).transpose().tocsc()

def concat_attr(adj, features, alpha=1.0):
    adj_sum = sp.csc_matrix(np.diag(np.array(adj.sum(axis=1)).squeeze()))
    features = normalize(features, norm='l1', axis=1)
    features = adj_sum @ features * alpha
    zeros = sp.csc_matrix(np.zeros(shape=(features.shape[1], features.shape[1])))
    
    return concatenate_sparse_matrices_by_rows(concatenate_csc_matrices_by_columns(adj, features), concatenate_csc_matrices_by_columns(features.transpose().tocsc(), zeros))

def normalize_trans(adj):
    # symmetric adj
    # output: sym-normed_adj, deg ** (-0.5)
    vol = adj.sum()
    adj_deg = np.array(adj.sum(axis=0)).squeeze()
    adj_deg_inv = sp.csc_matrix(np.diag(np.where(adj_deg>0, adj_deg**(-0.5), 0)))
    adj_normed = adj_deg_inv @ adj @ adj_deg_inv
    return adj_normed, adj_deg_inv.todense(), vol

# dataset split

k = args.proportion # used proportion of training set for evaluation
nb_nodes = adj.shape[0]
nb_train = int(nb_nodes * 0.8)
nb_train_real = int(nb_train * k)
nb_val_start = int(nb_train_real * 0.9)
random_perm = np.random.permutation(nb_nodes)
train_mask_real = random_perm[:nb_val_start]
val_mask = random_perm[nb_val_start:nb_train_real]
test_mask = random_perm[nb_train:]

def micro_f1(logits, labels):
    # Compute predictions
    preds = torch.argmax(logits, dim=1)
    
    return metrics.f1_score(labels, preds, average='micro')

def macro_f1(logits, labels):
    # Compute predictions
    preds = torch.argmax(logits, dim=1)
    
    return metrics.f1_score(labels, preds, average='macro')

# train

def train(model, optimizer, loss_func, adj, x, labels, train_mask, val_mask, test_mask, batch_size, epochs, iters_per_epoch, patience):
    save_name = 'save_model/' + str(int(time.time())) + str(args.proportion) + '_' + str(args.seed) + '.pkl'
    best_val_loss = 100 # a large loss
    p = 0
    for i in range(epochs):
        train_mask = train_mask[np.random.permutation(train_mask.shape[0])]
        train_start = 0
        acum_loss = 0
        acum_acc = 0
        for j in range(iters_per_epoch):
            model.train()
            if train_start + batch_size < train_mask.shape[0]:
                train_nodes = train_mask[train_start:train_start+batch_size]
                train_start += batch_size
            else:
                train_nodes = list(train_mask[train_start:]) + list(train_mask[:train_start+batch_size-train_mask.shape[0]])
                train_start += batch_size-train_mask.shape[0]
            
            output = model(adj, x)
            loss = loss_func(output[train_nodes], labels[train_nodes])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acum_loss += loss.detach().cpu()
            acum_acc += sum(torch.argmax(output[train_nodes], dim=1) == labels[train_nodes]).float()/batch_size
        
        with torch.no_grad():
            model.eval()
            output = model(adj, x)
            val_loss = loss_func(output[val_mask], labels[val_mask])
            val_acc = torch.sum(torch.argmax(output[val_mask], dim=1) == labels[val_mask]).float()/val_mask.shape[0]
            print('epoch: {}, training loss: {}, training acc: {}, val loss: {}, val acc: {}'.format(i, acum_loss/iters_per_epoch, acum_acc/iters_per_epoch, \
                                                                                                    val_loss, val_acc))
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_name)
                p = 0
            else:
                if p >= patience:
                    break
                p += 1
    
    with torch.no_grad():
        model.load_state_dict(torch.load(save_name))
        model.eval()
        output = model(adj, x)
        #test_acc = torch.sum(torch.argmax(output[test_mask], dim=1) == labels[test_mask]).float()/test_mask.shape[0]
        test_results = [micro_f1(output[test_mask].cpu(), labels[test_mask].cpu()), macro_f1(output[test_mask].cpu(), labels[test_mask].cpu())]
        print(test_results)
        output_csv(output_file, test_results)    
    os.remove(save_name)

#train mlp

def train_mlp(model, optimizer, loss_func, x, labels, train_mask, val_mask, test_mask, batch_size, epochs, iters_per_epoch, patience):
    save_name = 'save_model/' + str(int(time.time())) + str(args.proportion) + '_' + str(args.seed) + '.pkl'
    best_val_loss = 100 # a large loss
    for i in range(epochs):
        train_mask = train_mask[np.random.permutation(train_mask.shape[0])]
        train_start = 0
        acum_loss = 0
        acum_acc = 0
        for j in range(iters_per_epoch):
            model.train()
            if train_start + batch_size < train_mask.shape[0]:
                train_nodes = train_mask[train_start:train_start+batch_size]
                train_start += batch_size
            else:
                train_nodes = list(train_mask[train_start:]) + list(train_mask[:train_start+batch_size-train_mask.shape[0]])
                train_start += batch_size-train_mask.shape[0]
            
            output = model(x)
            loss = loss_func(output[train_nodes], labels[train_nodes])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acum_loss += loss.detach().cpu()
            acum_acc += sum(torch.argmax(output[train_nodes], dim=1) == labels[train_nodes]).float()/batch_size
        
        with torch.no_grad():
            model.eval()
            output = model(x)
            val_loss = loss_func(output[val_mask], labels[val_mask])
            val_acc = torch.sum(torch.argmax(output[val_mask], dim=1) == labels[val_mask]).float()/val_mask.shape[0]
            print('epoch: {}, training loss: {}, training acc: {}, val loss: {}, val acc: {}'.format(i, acum_loss/iters_per_epoch, acum_acc/iters_per_epoch, \
                                                                                                    val_loss, val_acc))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_name)
                p = 0
            else:
                if p >= patience:
                    break
                p += 1
    
    with torch.no_grad():
        model.load_state_dict(torch.load(save_name))
        model.eval()
        output = model(x)
        #test_acc = torch.sum(torch.argmax(output[test_mask], dim=1) == labels[test_mask]).float()/test_mask.shape[0]
        test_results = [micro_f1(output[test_mask].cpu(), labels[test_mask].cpu()), macro_f1(output[test_mask].cpu(), labels[test_mask].cpu())]
        print(test_results)
        output_csv(output_file, test_results)    
    os.remove(save_name)

def train_rna_rw(model, optimizer, loss_func, sentencedic, features, labels, train_mask, val_mask, test_mask, batch_size, epochs, iters_per_epoch, patience):
    save_name = 'save_model/' + str(int(time.time())) + str(args.proportion) + '_' + str(args.seed) + '.pkl'
    best_val_loss = 1e6 # a large loss

    sentences_val = [[]]
    val_mask_batch = [[]]
    batch_start = 0
    p=0
    for i in val_mask:
        if batch_start >= batch_size:
            sentences_val.append([])
            val_mask_batch.append([])
            batch_start = 0
        sentences_val[-1].extend(sentencedic[i])
        val_mask_batch[-1].append(i)
        batch_start += 1

    sentences_test = [[]]
    test_mask_batch = [[]]
    batch_start = 0
    for i in test_mask:
        if batch_start >= batch_size:
            sentences_test.append([])
            test_mask_batch.append([])
            batch_start = 0
        sentences_test[-1].extend(sentencedic[i])
        test_mask_batch[-1].append(i)
        batch_start += 1

    for i in range(epochs):
        train_mask = train_mask[np.random.permutation(train_mask.shape[0])]
        train_start = 0
        acum_loss = 0
        acum_acc = 0
        for j in range(iters_per_epoch):
            model.train()
            if train_start + batch_size < train_mask.shape[0]:
                train_nodes = train_mask[train_start:train_start+batch_size]
                train_start += batch_size
            else:
                train_nodes = list(train_mask[train_start:]) + list(train_mask[:train_start+batch_size-train_mask.shape[0]])
                train_start += batch_size-train_mask.shape[0]

            sentences = []
            for k in train_nodes:
                sentences.extend(sentencedic[k])

            output = model(features[sentences])
            loss = loss_func(output, labels[train_nodes])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acum_loss += loss.detach().cpu()
            acum_acc += sum(torch.argmax(output, dim=1) == labels[train_nodes]).float()/batch_size
        
        with torch.no_grad():
            acum_val_acc = 0
            acum_val_loss = 0
            model.eval()
            for mask, paths in zip(val_mask_batch, sentences_val):
                output = model(features[paths])
                acum_val_loss += loss_func(output, labels[mask]).detach().cpu() * len(mask)
                acum_val_acc += torch.sum(torch.argmax(output, dim=1) == labels[mask]).float()
            print('epoch: {}, training loss: {}, training acc: {}, val loss: {}, val acc: {}'.format(i, acum_loss/iters_per_epoch, acum_acc/iters_per_epoch, \
                                                                                                    acum_val_loss/len(val_mask), acum_val_acc/len(val_mask)))

            if acum_val_loss < best_val_loss:
                best_val_loss = acum_val_loss
                torch.save(model.state_dict(), save_name)
                p = 0
            else:
                if p >= patience:
                    break
                p += 1

    with torch.no_grad():
        model.load_state_dict(torch.load(save_name))
        model.eval()
        outputs = []
        for mask, paths in zip(test_mask_batch, sentences_test):
            outputs.append(model(features[paths]))
        #test_acc = torch.sum(torch.argmax(output[test_mask], dim=1) == labels[test_mask]).float()/test_mask.shape[0]
        outputs = torch.cat(outputs).cpu()
        test_results = [micro_f1(outputs.cpu(), labels[test_mask].cpu()), macro_f1(outputs.cpu(), labels[test_mask].cpu())]
        print(test_results)
        output_csv(output_file, test_results)
    
    os.remove(save_name)
                                                                                                    
# preprocess features

def preprocess_features(features):
    """Row-normalize feature matrix"""
    features = normalize(features, norm='l1', axis=1)
    return features.todense()

# numpy sparse to pytorch sparse

def np_sparse_to_pt_sparse(matrix):
    coo = matrix.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

if args.model == 'GCN':
    #x = preprocess_features(features)
    x = features.todense()
    x = torch.FloatTensor(x).to(device)
    labels = mat_contents['Label']
    labels = torch.LongTensor(labels-1).squeeze().to(device)
    #adj_pt, _, _ = normalize_trans(adj+sp.eye(adj.shape[0]))
    adj_pt = normalize_trans(adj)[0] + sp.eye(adj.shape[0])
    adj_pt = np_sparse_to_pt_sparse(adj_pt).to(device)

    model = GCN(x.size()[-1], 128, nb_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    loss_func = nn.CrossEntropyLoss()
    train(model, optimizer, loss_func, adj_pt, x, labels, train_mask_real, val_mask, test_mask, batch_size=128, epochs = 200, iters_per_epoch=int(nb_train_real/128)+1, patience=10)

elif args.model == 'GFNN' or args.model == 'SGC':
    x = preprocess_features(features)
    x = torch.FloatTensor(x).to(device)
    labels = mat_contents['Label']
    labels = torch.LongTensor(labels-1).squeeze().to(device)
    adj_pt, _, _ = normalize_trans(adj+sp.eye(adj.shape[0]))
    adj_pt = np_sparse_to_pt_sparse(adj_pt).to(device)
    x = torch.spmm(adj_pt, x)
    x = torch.spmm(adj_pt, x)

    if args.model == 'GFNN':
        model = MLP(x.size()[-1], 128, nb_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
    else:
        model = MLP(x.size()[-1], out_dim = nb_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    loss_func = nn.CrossEntropyLoss()
    train_mlp(model, optimizer, loss_func, x, labels, train_mask_real, val_mask, test_mask, batch_size=128, epochs = 200, iters_per_epoch=int(nb_train_real/128)+1, patience=10)

elif args.model == 'ATTR_RW_MF':
    ## ``Ours 1`` with (args.times_features == False)
    ## ``Ours 2`` with (args.times_features == True)

    if args.saved == False:
        # For the abalation study of local, non-local
        if args.subsample:
            features = subsample(adj, features)

        trans_attr_rw, trans_deg_inv, vol = normalize_trans(concat_attr(adj, features))

        rank_k = 300
        window_size = 5
        '''
        vals, vecs = LA.eigsh(trans_attr_rw, k=rank_k)

        vals_power = [vals]
        for i in range(window_size):
            vals_power.append(vals_power[-1] * vals)

        vals_power = sum(vals_power) / window_size
        trans_power = vecs @ np.diag(vals_power) @ vecs.transpose()
        '''
        trans_attr_rw = trans_attr_rw.todense()
        trans_power = [trans_attr_rw]
        for i in range(window_size-1):
            trans_power.append(trans_power[-1] @ trans_attr_rw)

        trans_power = sum(trans_power) / window_size
        mf = (trans_deg_inv @ trans_power @ trans_deg_inv * vol).real
        mf = mf[:nb_nodes, :nb_nodes]
        mf[mf<1] = 1
        mf = np.log(mf)
        np.save(args.dataset + '_attr_rw_win5_nosvd.npy', mf)
        exit()
    
    else:
        mf = np.load(args.dataset + '_attr_rw_win5_nosvd.npy')


    mf_sp = csc_matrix(mf)
    u, s, vt = LA.svds(mf_sp, k=200)

    mf_embed = u @ np.diag(s ** 0.5)
    
    if args.times_features:
        mf = mf[:nb_nodes, :nb_nodes] @ features
    else:
        mf = mf[:nb_nodes, :nb_nodes]

    x = torch.FloatTensor(mf).to(device)
    labels = mat_contents['Label']
    labels = torch.LongTensor(labels-1).squeeze().to(device)

    #model = MLP(x.size()[-1], hd_dim = 800, out_dim = nb_classes).to(device)
    model = MLP(x.size()[-1], out_dim = nb_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    loss_func = nn.CrossEntropyLoss()
    train_mlp(model, optimizer, loss_func, x, labels, train_mask_real, val_mask, test_mask, batch_size=128, epochs = 200, iters_per_epoch=int(nb_train_real/128)+1, patience=10)

elif args.model == 'ATTR_RW_MF_3':
    ## ``Ours 3`` in paper

    if args.saved == False:
        trans_attr_rw, trans_deg_inv, vol = normalize_trans(concat_attr(adj, features))

        rank_k = 200
        window_size = 10
        vals, vecs = LA.eigsh(trans_attr_rw, k=rank_k)

        vals_power = [vals]
        for i in range(window_size):
            vals_power.append(vals_power[-1] * vals)

        vals_power = sum(vals_power) / window_size
        trans_power = vecs @ np.diag(vals_power) @ vecs.transpose()
        mf = (trans_deg_inv @ trans_power @ trans_deg_inv * vol).real
        mf = mf[:nb_nodes, :nb_nodes]
        mf[mf<1] = 1
        mf = np.log(mf)

    else:
        mf = np.load(args.dataset + '_attr_rw_win5_nosvd.npy')

    mf_sp = csc_matrix(mf)
    u, s, vt = LA.svds(mf_sp, k=200)

    mf_embed = u @ np.diag(s ** 0.5)

    x = torch.FloatTensor(mf_embed).to(device)
    labels = mat_contents['Label']
    labels = torch.LongTensor(labels-1).squeeze().to(device)

    model = MLP(x.size()[-1], out_dim = nb_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    loss_func = nn.CrossEntropyLoss()
    train_mlp(model, optimizer, loss_func, x, labels, train_mask_real, val_mask, test_mask, batch_size=128, epochs = 200, iters_per_epoch=int(nb_train_real/128)+1, patience=10)

elif args.model == 'NetMF':
    ## ``NetMF`` in paper
    `
    trans_attr_rw, trans_deg_inv, vol = normalize_trans(adj)

    rank_k = 200
    window_size = 5
    #vals, vecs = LA.eigsh(trans_attr_rw, k=rank_k)

    #vals_power = [vals]
    trans_attr_rw = trans_attr_rw.todense()
    trans_power = [trans_attr_rw]
    for i in range(window_size):
    #    vals_power.append(vals_power[-1] * vals)
        trans_power.append(trans_power[-1] @ trans_attr_rw)

    #vals_power = sum(vals_power) / window_size
    #trans_power = vecs @ np.diag(vals_power) @ vecs.transpose()
    trans_power = sum(trans_power) / window_size
    mf = (trans_deg_inv @ trans_power @ trans_deg_inv * vol).real
    mf[mf<1] = 1
    mf = np.log(mf)

    mf_sp = csc_matrix(mf)
    u, s, vt = LA.svds(mf_sp, k=200)

    mf_embed = u @ np.diag(s ** 0.5)

    x = torch.FloatTensor(mf_embed).to(device)
    labels = mat_contents['Label']
    labels = torch.LongTensor(labels-1).squeeze().to(device)

    model = MLP(x.size()[-1], out_dim = nb_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    loss_func = nn.CrossEntropyLoss()
    train_mlp(model, optimizer, loss_func, x, labels, train_mask_real, val_mask, test_mask, batch_size=128, epochs = 200, iters_per_epoch=int(nb_train_real/128)+1, patience=10)

elif args.model == 'GRAPHRNA_RW':
    ## ``AttrRW`` in paper

    sentencedic, sentnumdic = walk_dic_featwalk(adj, features, num_paths=100, path_length=5, alpha=0.5).function()

    model = pro_lstm_featwalk(nfeat=features.shape[1], nhid=200, nclass=nb_classes, dropout=0.5, num_paths=100, path_length=5).to(device)

    parameter = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameter, lr=0.0001, weight_decay=5e-4)
    features = torch.FloatTensor(features.todense()).to(device)
    features = torch.cat((features, torch.eye(features.shape[1]).to(device)), 0)

    labels = mat_contents['Label']
    labels = torch.LongTensor(labels-1).squeeze().to(device)

    loss_func = nn.CrossEntropyLoss()
    train_rna_rw(model, optimizer, loss_func, sentencedic, features, labels, train_mask_real, val_mask, test_mask, batch_size=32, epochs=200, iters_per_epoch=int(nb_train_real/32)+1, patience=10)

elif args.model == 'GRAPHRNA_RW_FULL':
    ## ``AttrRW + RNN`` in paper

    sentencedic, sentnumdic = walk_dic_featwalk(adj, features, num_paths=100, path_length=5, alpha=0.5).function()

    model = pro_lstm_featwalk_full(nfeat=features.shape[1], nhid=200, nclass=nb_classes, dropout=0.5, num_paths=100, path_length=5).to(device)

    parameter = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameter, lr=0.0001, weight_decay=5e-4)
    features = torch.FloatTensor(features.todense()).to(device)
    features = torch.cat((features, torch.eye(features.shape[1]).to(device)), 0)

    labels = mat_contents['Label']
    labels = torch.LongTensor(labels-1).squeeze().to(device)

    loss_func = nn.CrossEntropyLoss()
    train_rna_rw(model, optimizer, loss_func, sentencedic, features, labels, train_mask_real, val_mask, test_mask, batch_size=32, epochs=200, iters_per_epoch=int(nb_train_real/32)+1, patience=10)
