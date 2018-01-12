#!/usr/bin/env python

import argparse
import os
import os.path as osp

import fcn
import numpy as np
import skimage.io
import torch
from torch.autograd import Variable
import torchfcn
import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', help='Model path')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-e', '--pixel_embeddings', type=int, default=-1)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    model_file = args.model_file
    if args.pixel_embeddings != -1:
        pixel_embeddings = args.pixel_embeddings
    else:
        pixel_embeddings = None
 
    root = '/opt/visualai/rkdoshi/pytorch-fcn/data/datasets'
    val_loader = torch.utils.data.DataLoader(
        torchfcn.datasets.VOC2011ClassSeg(
            root, split='seg11valid', transform=True, pixel_embeddings=pixel_embeddings),
        batch_size=1, shuffle=False,
        num_workers=8, pin_memory=True)

    n_class = len(val_loader.dataset.class_names)

    if pixel_embeddings:
        model = torchfcn.models.FCN32s(n_class=pixel_embeddings)
    else:
        model = torchfcn.models.FCN32s(n_class=21)
    
    if torch.cuda.is_available():
        model = model.cuda()
    print('==> Loading %s model file: %s' %
          (model.__class__.__name__, model_file))
    model_data = torch.load(model_file)
    try:
        model.load_state_dict(model_data)
    except Exception:
        model.load_state_dict(model_data['model_state_dict'])
    model.eval()


    if pixel_embeddings:
        embed_arr = torchfcn.utils.load_obj('/opt/visualai/rkdoshi/pytorch-fcn/examples/voc/label2vec_dict_' + str(pixel_embeddings))
        embed_arr = torch.from_numpy(embed_arr).cuda().float()

    print('==> Evaluating with VOC2011ClassSeg seg11valid')
    visualizations = []
    label_trues, label_preds = [], []
    for batch_idx, (data, target) in tqdm.tqdm(enumerate(val_loader),
                                               total=len(val_loader),
                                               ncols=80, leave=False):
        if pixel_embeddings:
            target, target_embed = target
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        if pixel_embeddings:
            target_embed = Variable(target_embed.cuda())
        score = model(data)

        imgs = data.data.cpu()
        if pixel_embeddings:
           lbl_pred = torchfcn.utils.get_lbl_pred(score, embed_arr)
        else:
            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
        lbl_true = target.data.cpu()
        
        
        # if batch_idx == 1:
        #     num_true_bk = lbl_true[lbl_true == 0].sum()
        #     num_pred_bk = np.sum(lbl_pred[lbl_pred == 0])
        #     raise Exception(score.sum())
    
        for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
            img, lt = val_loader.dataset.untransform(img, lt)
            label_trues.append(lt)
            label_preds.append(lp)
            if len(visualizations) < 9:
                viz = fcn.utils.visualize_segmentation(
                    lbl_pred=lp, lbl_true=lt, img=img, n_class=n_class,
                    label_names=val_loader.dataset.class_names)
                visualizations.append(viz)
    metrics = torchfcn.utils.label_accuracy_score(
        label_trues, label_preds, n_class=n_class)
    metrics = np.array(metrics)
    metrics *= 100
    print('''\
Accuracy: {0}
Accuracy Class: {1}
Mean IU: {2}
FWAV Accuracy: {3}'''.format(*metrics))

    viz = fcn.utils.get_tile_image(visualizations)
    skimage.io.imsave('viz_evaluate.png', viz)


if __name__ == '__main__':
    main()
