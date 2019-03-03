import torch as th
from torch.utils.data import Dataset
import numpy as np 
import os    
import math    
import random
import pickle

class MSVD(Dataset):
    """MSVD dataset."""

    def __init__(self, visual_features, text_features, sequence_features,
            train_list, test_list, 
            coco=False, max_words=30,verbose=False):
        """
        Args:
        """
        self.max_words = max_words
        print 'loading data ...'

        with open(train_list) as f:
            self.train_list = f.readlines()

        self.train_list = [x.strip() for x in self.train_list]

        with open(test_list) as f:
            self.test_list = f.readlines()

        self.test_list = [x.strip() for x in self.test_list]


        pickle_in = open(visual_features,'rb')
        self.visual_features = pickle.load(pickle_in)

        pickle_in = open(sequence_features,'rb')
        self.sequence_features = pickle.load(pickle_in)

        pickle_in = open(text_features,'rb')
        self.text_features = pickle.load(pickle_in)

        self.coco = coco

        self.n_MSR = len(self.train_list)
        self.coco_ind = np.zeros((self.n_MSR))
        self.n_coco = 0
 

        # computing retrieval

        self.video_retrieval = np.zeros((len(self.test_list),2048))
        self.sequence_retrieval = np.zeros((len(self.test_list), max_words, 2048))
        self.text_retrieval = np.zeros((len(self.test_list), max_words, 300))
        self.face_ind_retrieval = np.ones((len(self.test_list)))
        
        for i in range(len(self.test_list)):
            self.video_retrieval[i] = self.visual_features[self.test_list[i]]
            self.face_ind_retrieval[i] = 0

            la = len(self.sequence_features[self.test_list[i]])
            self.sequence_retrieval[i,:min(max_words,la),:] = self.sequence_features[self.test_list[i]][:min(max_words,la)]
            
            lt = len(self.text_features[self.test_list[i]][0])
            self.text_retrieval[i,:min(max_words,lt),:] = self.text_features[self.test_list[i]][0][:min(max_words,lt)]

        
        self.video_retrieval = th.from_numpy(self.video_retrieval).float()
        self.sequence_retrieval = th.from_numpy(self.sequence_retrieval).float()
        self.text_retrieval = th.from_numpy(self.text_retrieval).float()
        
        print 'done'

    def collate_data(self, data):
        video_tensor = np.zeros((len(data), 2048))
        sequence_tensor = np.zeros((len(data), self.max_words,2048))
        text_tensor = np.zeros((len(data), self.max_words, 300))
        coco_ind = np.zeros((len(data)))
        face_ind = np.zeros((len(data)))

        for i in range(len(data)):

            coco_ind[i] = data[i]['coco_ind']
            video_tensor[i] = data[i]['video']
            
            la = len(data[i]['sequence'])
            sequence_tensor[i,:min(la,self.max_words), :] = data[i]['sequence'][:min(self.max_words,la)]

            lt = len(data[i]['text'])
            text_tensor[i,:min(lt,self.max_words), :] = data[i]['text'][:min(self.max_words,lt)]


        return {'video': th.from_numpy(video_tensor).float(),
                'coco_ind': coco_ind,
                'face_ind': face_ind,
                'text': th.from_numpy(text_tensor).float(),
                'sequence': th.from_numpy(sequence_tensor).float()
                }


    def __len__(self):
        return len(self.coco_ind)

    def __getitem__(self, idx):

        face_ind = 1
        if idx < self.n_MSR:
            vid = self.train_list[idx]
            text = self.text_features[vid]
            # r = random.randint(0, len(text)-1)
            text = text[0]
            sequence = self.sequence_features[vid]
            video = self.visual_features[vid]

        elif self.coco:
            video = self.coco_visual[idx-self.n_MSR]
            text = self.coco_text[idx-self.n_MSR]
            sequence = th.zeros(1,2048)
            face = th.zeros(128)
            face_ind = 0

        return {'video': video, 
                'text': text,
                'coco_ind': self.coco_ind[idx],
                'face_ind': face_ind,
                'sequence': sequence
                }

    def getRetrievalSamples(self):
        return {'video': self.video_retrieval, 
                'text': self.text_retrieval,
                'face_ind': self.face_ind_retrieval,
                'sequence': self.sequence_retrieval
                }

