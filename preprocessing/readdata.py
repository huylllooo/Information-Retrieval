import torch as th
from torch.utils.data import Dataset, DataLoader
import LSMDC as LD2
import MSRVTT as MSR
import numpy as np
import torch.optim as optim
import argparse
from loss import MaxMarginRankingLoss
from model import Net
from torch.autograd import Variable
import os
import random
import pickle
from qcm_sampler import QCMSampler
from MSR_sampler import MSRSampler

root_feat = 'data'

# predefining random initial seeds
# th.manual_seed(args.seed)
# np.random.seed(args.seed)
# random.seed(args.seed)

# print 'Reading features ...'

# visual_feat_path = os.path.join(root_feat,'frameLevel-resnet152.pickle')
# pickle_in = open(visual_feat_path,'rb')
# visual_features = pickle.load(pickle_in)
# print type(visual_features)
# print visual_features['R0e8ojL0vcc_8_14'].shape

audio_feat_path = os.path.join(root_feat,'audio_features.pickle')
pickle_in = open(audio_feat_path,'rb')
audio_features = pickle.load(pickle_in)
print type(audio_features)
print audio_features['video1027'].shape

# text_feat_path = os.path.join(root_feat,'w2v_MSRVTT.pickle')
# pickle_in = open(text_feat_path,'rb')
# text_features = pickle.load(pickle_in)
# print type(text_features)
# print len(text_features['video1027'])
# print text_features['video1027'][3]
# print text_features['video1027'][5].shape
# print type(text_features['video1027'][3])


# with open('captionGloVe.pickle', 'rb') as handle:
#     captionList = pickle.load(handle)

# print len(captionList)

# with open('frameLevel-resnet152.pickle', 'rb') as handle:
#     vidList = pickle.load(handle)

# print len(vidList)
# i = 0
# for key, value in vidList.items():
# 	if key not in captionList.keys():
# 		del vidList[key]
# print len(vidList)

# with open('frameLevel-resnet152.pickle', 'wb') as handle:
#     pickle.dump(vidList, handle, protocol=pickle.HIGHEST_PROTOCOL)
# if args.MSRVTT:
#     visual_feat_path = os.path.join(root_feat,'resnet_features.pickle')  
#     # flow_feat_path = os.path.join(root_feat,'flow_features.pickle')
#     text_feat_path = os.path.join(root_feat,'w2v_MSRVTT.pickle')
#     audio_feat_path = os.path.join(root_feat,'audio_features.pickle')
#     # face_feat_path = os.path.join(root_feat,'face_features.pickle')

#     train_list_path = os.path.join(root_feat,'train_list.txt')
#     test_list_path = os.path.join(root_feat,'test_list.txt')

#     dataset = MSR.MSRVTT(visual_feat_path, text_feat_path, audio_feat_path, train_list_path,test_list_path, coco=args.coco) 
#     msr_sampler = MSRSampler(dataset.n_MSR, dataset.n_coco, args.coco_sampling_rate)
    
#     if args.coco:
#         dataloader = DataLoader(dataset, batch_size=args.batch_size,
#                 sampler=msr_sampler, num_workers=1,collate_fn=dataset.collate_data, drop_last=True)
#     else:
#         dataloader = DataLoader(dataset, batch_size=args.batch_size,
#                 shuffle=True, num_workers=1,collate_fn=dataset.collate_data, drop_last=True)

# print 'Done.'