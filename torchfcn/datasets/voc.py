#!/usr/bin/env python

import collections
import os.path as osp

import numpy as np
import PIL.Image
import scipy.io
import torch
from torch.utils import data
import pickle
import torchfcn

class VOCClassSegBase(data.Dataset):
    class_names = np.array([
        'background',
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'potted plant',
        'sheep',
        'sofa',
        'train',
        'tv/monitor',
    ])
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

    def __init__(self, root, split='train', transform=False, pixel_embeddings=None):
        self.root = root
        self.split = split
        self._transform = transform
        self.pixel_embeddings = pixel_embeddings

        if self.pixel_embeddings:
            embed_dim = pixel_embeddings # dimensions in each embedding
            embeddings_dict = torchfcn.utils.load_obj('/opt/visualai/rkdoshi/pytorch-fcn/examples/voc/label2vec_dict_' + str(embed_dim))
            num_embed = embeddings_dict.shape[0] #  21 = background (class 0) + labels (class 1-20)
            self.embeddings = torch.nn.Embedding(num_embed, embed_dim)
            self.embeddings.weight.requires_grad = False
            self.embeddings.weight.data.copy_(torch.from_numpy(embeddings_dict))

        # VOC2011 and others are subset of VOC2012
        dataset_dir = osp.join(self.root, 'VOC/VOCdevkit/VOC2012')
        self.files = collections.defaultdict(list)
        for split in ['train', 'val']:
            imgsets_file = osp.join(dataset_dir, 'ImageSets/Segmentation/%s.txt' % split)
            for did in open(imgsets_file):
                did = did.strip()
                img_file = osp.join(dataset_dir, 'JPEGImages/%s.jpg' % did)
                lbl_file = osp.join(dataset_dir, 'SegmentationClass/%s.png' % did)
                self.files[split].append({'img': img_file, 'lbl': lbl_file,})

    def __len__(self):          
        return len(self.files[self.split])

    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        # load image
        img_file = data_file['img']
        img = PIL.Image.open(img_file)
        img = np.array(img, dtype=np.uint8)
        # load label
        lbl_file = data_file['lbl']
        lbl = PIL.Image.open(lbl_file)
        lbl = np.array(lbl, dtype=np.int32)
        lbl[lbl == 255] = -1
        if self.pixel_embeddings:
            # change to  0 instead of -1 because embedding lookup cannot handle index -1
            lbl_cpy = np.copy(lbl)
            lbl_cpy[lbl_cpy == -1] = 0 
            lbl_vec = self.embeddings(torch.from_numpy(lbl_cpy).long()).data

        if self._transform:
            img, lbl = self.transform(img, lbl)
                
        if self.pixel_embeddings:
            return img, (lbl, lbl_vec)
        else:
            return img, lbl

    def transform(self, img, lbl):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def untransform(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        lbl = lbl.numpy()
        return img, lbl


class VOC2011ClassSeg(VOCClassSegBase):

    def __init__(self, root, split='train', transform=False, pixel_embeddings=None):
        super(VOC2011ClassSeg, self).__init__(
            root, split=split, transform=transform, pixel_embeddings=pixel_embeddings)
        if self.pixel_embeddings:
            embeddings_dict = torchfcn.utils.load_obj('/opt/visualai/rkdoshi/pytorch-fcn/examples/voc/label2vec_dict_' + str(pixel_embeddings))
        pkg_root = osp.join(osp.dirname(osp.realpath(__file__)), '..')
        imgsets_file = osp.join(pkg_root, 'ext/fcn.berkeleyvision.org', 'data/pascal/seg11valid.txt')
        dataset_dir = osp.join(self.root, 'VOC/VOCdevkit/VOC2012')
        for did in open(imgsets_file):
            did = did.strip()
            img_file = osp.join(dataset_dir, 'JPEGImages/%s.jpg' % did)
            lbl_file = osp.join(dataset_dir, 'SegmentationClass/%s.png' % did)
            self.files['seg11valid'].append({'img': img_file, 'lbl': lbl_file})


class VOC2012ClassSeg(VOCClassSegBase):

    url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'  # NOQA

    def __init__(self, root, split='train', transform=False):
        super(VOC2012ClassSeg, self).__init__(root, split=split, transform=transform)


class SBDClassSeg(VOCClassSegBase):

    # XXX: It must be renamed to benchmark.tar to be extracted.
    url = 'http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz'  # NOQA

    # pixel_embeddings is set to the number of dimensions for the pixel-embeddings
    def __init__(self, root, split='train', transform=False, pixel_embeddings=None):
        self.root = root
        self.split = split
        self._transform = transform
        self.pixel_embeddings = pixel_embeddings
        if self.pixel_embeddings:   
            embed_dim = pixel_embeddings # dimensions in each embedding
            embeddings_dict = torchfcn.utils.load_obj('/opt/visualai/rkdoshi/pytorch-fcn/examples/voc/label2vec_dict_' + str(embed_dim))
            num_embed = embeddings_dict.shape[0] #  21 = background (class 0) + labels (class 1-20)
            self.embeddings = torch.nn.Embedding(num_embed, embed_dim)
            self.embeddings.weight.requires_grad = False
            self.embeddings.weight.data.copy_(torch.from_numpy(embeddings_dict))

        dataset_dir = osp.join(self.root, 'VOC/benchmark_RELEASE/dataset')
        self.files = collections.defaultdict(list)
        for split in ['train', 'val']:
            imgsets_file = osp.join(dataset_dir, '%s.txt' % split)
            for did in open(imgsets_file):
                did = did.strip()
                img_file = osp.join(dataset_dir, 'img/%s.jpg' % did)
                lbl_file = osp.join(dataset_dir, 'cls/%s.mat' % did)
                self.files[split].append({'img': img_file, 'lbl': lbl_file,})

    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        # load image
        img_file = data_file['img']
        img = PIL.Image.open(img_file)
        img = np.array(img, dtype=np.uint8)
        # load label
        lbl_file = data_file['lbl']
        mat = scipy.io.loadmat(lbl_file)
        lbl = mat['GTcls'][0]['Segmentation'][0].astype(np.int32)   
        lbl[lbl == 255] = -1

        if self.pixel_embeddings:
            lbl_cpy = np.copy(lbl)
            lbl_cpy[lbl_cpy == -1] = 0
            lbl_vec = self.embeddings(torch.from_numpy(lbl_cpy).long()).data
        
        if self._transform:
            img, lbl = self.transform(img, lbl)
        
        if self.pixel_embeddings:
            return img, (lbl, lbl_vec)
        else:
            return img, lbl
