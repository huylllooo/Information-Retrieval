import torch as th
from torch.utils.data import Dataset, DataLoader
import MSVD
import numpy as np
import torch.optim as optim
import argparse
from loss import MaxMarginRankingLoss
from model import Net
from torch.autograd import Variable
import os
import random


parser = argparse.ArgumentParser(description='MSVD-IR')

parser.add_argument('--lr', type=float, default=0.0004,
                            help='initial learning rate')
parser.add_argument('--epochs', type=int, default=50,
                            help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=128,
                            help='batch size')
parser.add_argument('--text_cluster_size', type=int, default=32,
                            help='Text cluster size')
parser.add_argument('--margin', type=float, default=0.2,
                            help='MaxMargin margin value')
parser.add_argument('--lr_decay', type=float, default=0.95,
                            help='Learning rate exp epoch decay')
parser.add_argument('--n_display', type=int, default=100,
                            help='Information display frequence')
parser.add_argument('--GPU', type=bool, default=True,
                            help='Use of GPU')
parser.add_argument('--n_cpu', type=int, default=1,
                            help='Number of CPU')

parser.add_argument('--model_name', type=str, default='test',
                            help='Model name')
parser.add_argument('--seed', type=int, default=1,
                            help='Initial Random Seed')

parser.add_argument('--optimizer', type=str, default='adam',
                            help='optimizer')
parser.add_argument('--momentum', type=float, default=0.9,
                            help='Nesterov Momentum for SGD')

parser.add_argument('--coco_sampling_rate', type=float, default=1.0,
                            help='coco sampling rate')


args = parser.parse_args()

print args

root_feat = 'data'

def verbose(epoch, status, metrics, name='TEST'):
    print(name+' - epoch: %d, epoch status: %.2f, r@1: %.3f, r@5: %.3f, r@10: %.3f, mr: %d' % 
            (epoch + 1, status, 
                metrics['R1'], metrics['R5'], metrics['R10'],
                metrics['MR']))

def compute_metric(x):
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:,np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]

    metrics = {}
    metrics['R1'] = float(np.sum(ind == 0))/len(ind)
    metrics['R5'] = float(np.sum(ind < 5))/len(ind)
    metrics['R10'] = float(np.sum(ind < 10))/len(ind)
    metrics['MR'] = np.median(ind) + 1
    metrics['size'] = len(ind)

    return metrics

def make_tensor(l, max_len):
    tensor = np.zeros((len(l),max_len,l[0].shape[-1]))
    for i in range(len(l)):
        if len(l[i]):
            tensor[i,:min(max_len,l[i].shape[0]),:] = l[i][:min(max_len,l[i].shape[0])]

    return th.from_numpy(tensor).float()

# predefining random initial seeds
th.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

print 'Pre-loading features ... This may takes several minutes ...'

visual_feat_path = os.path.join(root_feat,'1-frame-resnet152.pickle')
text_feat_path = os.path.join(root_feat,'captionGloVe.pickle')
sequence_feat_path = os.path.join(root_feat,'frameLevel-resnet152.pickle')

train_list_path = os.path.join(root_feat,'train_list_MSVD.txt')
test_list_path = os.path.join(root_feat,'test_list_MSVD.txt')

dataset = MSVD.MSVD(visual_feat_path, text_feat_path,
        sequence_feat_path, train_list_path,test_list_path) 


dataloader = DataLoader(dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=0,collate_fn=dataset.collate_data, drop_last=True)

print 'Done.'

# Model
video_modality_dim = {'sequence': (2048*16, 2048),
'visual': (2048,2048)}
net = Net(video_modality_dim,300,
        sequence_cluster=16,text_cluster=args.text_cluster_size)
net.train()
if args.GPU:
    net.cuda()

# Optimizers + Loss
max_margin = MaxMarginRankingLoss(margin=args.margin) 

if args.optimizer == 'adam':
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
elif args.optimizer == 'sgd':
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)

if args.GPU:
    max_margin.cuda()

n_display = args.n_display
dataset_size = len(dataset)
lr_decay = args.lr_decay

print 'Starting training loop ...'

for epoch in range(args.epochs):
    running_loss = 0.0
    print 'epoch: %d'%epoch

    for i_batch, sample_batched in enumerate(dataloader):

        captions = sample_batched['text']
        sequence = sample_batched['sequence']
       

        video = sample_batched['video']
        coco_ind = sample_batched['coco_ind']
        face_ind = sample_batched['face_ind']

        ind = {}
        # ind['face'] = face_ind
        ind['visual'] = np.ones((len(face_ind)))
        ind['sequence'] = 1 - coco_ind

        if args.GPU:
            captions, video = Variable(captions.cuda()), Variable(video.cuda())
            sequence  =  Variable(sequence.cuda())
            # face = Variable(face.cuda())


        optimizer.zero_grad()
        confusion_matrix = net(captions,
                { 'sequence': sequence, 'visual': video}, ind, True)
        loss = max_margin(confusion_matrix)
        loss.backward()

        optimizer.step()
        running_loss += loss.data[0]
        
        if (i_batch+1) % n_display == 0:
            print 'Epoch %d, Epoch status: %.2f, Training loss: %.4f'%(epoch + 1,
                    args.batch_size*float(i_batch)/dataset_size,running_loss/n_display)
            running_loss = 0.0

    print 'evaluating epoch %d ...'%(epoch+1)
    net.eval()  

    retrieval_samples = dataset.getRetrievalSamples()

    video = Variable(retrieval_samples['video'].cuda(), volatile=True)
    captions = Variable(retrieval_samples['text'].cuda(), volatile=True)
    sequence = Variable(retrieval_samples['sequence'].cuda(), volatile=True)

    face_ind = retrieval_samples['face_ind']

    ind = {}
    ind['face'] = face_ind
    ind['visual'] = np.ones((len(face_ind)))
    ind['sequence'] = np.ones((len(face_ind)))

    conf = net(captions,
            {'sequence': sequence, 'visual': video}, ind, True)
    confusion_matrix = conf.data.cpu().float().numpy()
    metrics = compute_metric(confusion_matrix)

    print metrics['size']
    verbose(epoch, args.batch_size*float(i_batch)/dataset_size, metrics, name='MSVD')
        
    net.train()

    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
