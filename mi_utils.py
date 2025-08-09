import torch
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.metrics import roc_curve, auc
from torch.nn import CrossEntropyLoss
from torch.nn.functional import cross_entropy
from tqdm import tqdm
# from detect_gpt_utils import *
import timeit
from transformers import AutoTokenizer
from torch.autograd import Variable
# from deepspeed.utils import safe_get_full_grad
import psutil
import subprocess
from typing import Optional
from transformers.generation.utils import GenerationMixin, GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList

def plot_hist(
    train_statistic, val_statistic,
    title, log_scale=False,show_plot=True,
    save_name=None
):
    '''
    Plots histogram of train and validation test statistics.
    '''
    if torch.is_tensor(train_statistic):
        train_statistic = train_statistic.flatten()
    else:
        train_statistic = torch.tensor(train_statistic).flatten()
    train_statistic = train_statistic[~train_statistic.isnan()]
    if torch.is_tensor(val_statistic):
        val_statistic = val_statistic.flatten()
    else:
        val_statistic = torch.tensor(val_statistic).flatten()
    val_statistic = val_statistic[~val_statistic.isnan()]
    
    plt.hist(train_statistic, bins=50, alpha=0.5, label='train')
    plt.hist(val_statistic, bins=50, alpha=0.5, label='val')
    plt.title(title)
    plt.legend(loc='upper right')
    if save_name is not None:
        if "png" not in save_name:
            save_name = save_name + ".png"
        plt.savefig(save_name, bbox_inches="tight")
    
    plt.clf()
    
def plot_layer_auc_curve(
    layer_aucs, title, save_name=None
):
    '''
    Plots AUC curve of each layer.
    '''
    plt.plot(layer_aucs)
    plt.title(title)
    plt.xlabel('Layer')
    plt.ylabel('AUC')
    if save_name is not None:
        if "png" not in save_name:
            save_name = save_name + ".png"
        plt.savefig(save_name, bbox_inches="tight")
    # plt.show()
    plt.clf()
    

def plot_ROC(
    train_statistic, val_statistic,
    title, log_scale=False,show_plot=True,
    save_name=None):
    '''
    Plots ROC with train and validation test statistics. Note that we assume train statistic < test statistic. Negate before using if otherwise.
    '''
    if torch.is_tensor(train_statistic):
        train_statistic = train_statistic.flatten()
    else:
        train_statistic = torch.tensor(train_statistic).flatten()
    train_statistic = train_statistic[~train_statistic.isnan()]
    if torch.is_tensor(val_statistic):
        val_statistic = val_statistic.flatten()
    else:
        val_statistic = torch.tensor(val_statistic).flatten()
    val_statistic = val_statistic[~val_statistic.isnan()]

    fpr, tpr, thresholds = roc_curve(torch.cat((torch.ones_like(train_statistic),torch.zeros_like(val_statistic))).flatten(),
                                    torch.cat((-train_statistic,-val_statistic)).flatten())
    roc_auc = auc(fpr, tpr)
    if roc_auc < 0.5:
        fpr, tpr, thresholds = roc_curve(
            torch.cat((torch.zeros_like(train_statistic),torch.ones_like(val_statistic))).flatten(),
            torch.cat((-train_statistic,-val_statistic)).flatten()
        )
        roc_auc = auc(fpr, tpr)
    plt.figure()
    if not log_scale:
        plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.4f)' % roc_auc)
    else:
        plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.4f)' % roc_auc)
        plt.xscale("log",base=10,subs=list(range(11)))
        plt.yscale("log",base=10,subs=list(range(11)))
        # plt.xscale("symlog",base=10,subs=list(range(11)),linthresh=1e-3,linscale=0.25)
        # plt.yscale("symlog",base=10,subs=list(range(11)),linthresh=1e-3,linscale=0.25)
        plt.xlim(9e-4,1.1)
        plt.ylim(9e-4,1.1)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    print(f"AUC of Experiment {title}\n{roc_auc}")
    if save_name is not None:
        if "png" not in save_name:
            save_name = save_name + ".png"
        plt.savefig(save_name, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.clf()
    
def compute_AUC(train_statistic,val_statistic):
    '''
    Plots ROC with train and validation test statistics. Note that we assume train statistic < test statistic. Negate before using if otherwise.
    '''
    if torch.is_tensor(train_statistic):
        train_statistic = train_statistic.flatten()
    else:
        train_statistic = torch.tensor(train_statistic).flatten()
    train_statistic = train_statistic[~train_statistic.isnan()]
    if torch.is_tensor(val_statistic):
        val_statistic = val_statistic.flatten()
    else:
        val_statistic = torch.tensor(val_statistic).flatten()
    val_statistic = val_statistic[~val_statistic.isnan()]

    fpr, tpr, thresholds = roc_curve(torch.cat((torch.ones_like(train_statistic),torch.zeros_like(val_statistic))).flatten(),
                                    torch.cat((-train_statistic,-val_statistic)).flatten())
    roc_auc = auc(fpr, tpr)
    if roc_auc < 0.5:
        fpr, tpr, thresholds = roc_curve(
            torch.cat((torch.zeros_like(train_statistic),torch.ones_like(val_statistic))).flatten(),
            torch.cat((-train_statistic,-val_statistic)).flatten()
        )
        roc_auc = auc(fpr, tpr)
    return roc_auc

def compute_tpr_25_05_01_001_0(train_statistic,val_statistic):
    '''
    Plots ROC with train and validation test statistics. Note that we assume train statistic < test statistic. Negate before using if otherwise.
    '''
    if torch.is_tensor(train_statistic):
        train_statistic = train_statistic.flatten()
    else:
        train_statistic = torch.tensor(train_statistic).flatten()
    train_statistic = train_statistic[~train_statistic.isnan()]
    if torch.is_tensor(val_statistic):
        val_statistic = val_statistic.flatten()
    else:
        val_statistic = torch.tensor(val_statistic).flatten()
    val_statistic = val_statistic[~val_statistic.isnan()]

    fpr, tpr, thresholds = roc_curve(torch.cat((torch.ones_like(train_statistic),torch.zeros_like(val_statistic))).flatten(),
                                    torch.cat((-train_statistic,-val_statistic)).flatten())
    # roc_auc = auc(fpr, tpr)
    # if roc_auc < 0.5:
    #     fpr, tpr, thresholds = roc_curve(
    #         torch.cat((torch.zeros_like(train_statistic),torch.ones_like(val_statistic))).flatten(),
    #         torch.cat((-train_statistic,-val_statistic)).flatten()
    #     )
    #     roc_auc = auc(fpr, tpr)
    tpr_25 = tpr[np.max(np.argwhere(fpr<=0.25))]
    tpr_05 = tpr[np.max(np.argwhere(fpr<=0.05))]
    tpr_01 = tpr[np.max(np.argwhere(fpr<=0.01))]
    tpr_001 = tpr[np.max(np.argwhere(fpr<=0.001))]
    tpr_0 = tpr[np.max(np.argwhere(fpr<=0))]
    return tpr_25, tpr_05, tpr_01, tpr_001, tpr_0

