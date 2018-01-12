import numpy as np
import pickle

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.

      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc

# get nearest label prediction for pixel embeddings of size (n,c, h, w) 
# score: torch.Size([1, 50, 366, 500])
def get_lbl_pred(score, embed_arr):
   n, c, h, w = score.size()
   n_classes = embed_arr.shape[0]
   embeddings = embed_arr.transpose(1,0).repeat(1,h*w,1,1)
   score = score.view(1,h*w,c,1).repeat(1,1,1,n_classes)
   dist = score.data - embeddings
   dist = dist.pow(2).sum(2).sqrt()
   min_val, indices = dist.min(2)
   return indices.view(1,h,w).cpu().numpy()

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f, encoding='latin-1')
