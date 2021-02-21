from conf import *

import gc
import os
import argparse
import sys
import time
import cv2
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

import torch
import torch.nn as nn
from torchvision import transforms
from warmup_scheduler import GradualWarmupScheduler

from dataloader import *
from models import *
from trainer import *
from transforms import *
from optimizer import *
from utils import *

import warnings
warnings.filterwarnings('ignore')

from glob import glob


def create_str_feature(df):
    df = df.rename(columns={0:'patient', 1:'image', 2:'label'})
    df['time'] = df['image'].apply(lambda x : int(x.split('_')[-1][:-4]))
    df['user_count'] = df.patient.map(df.groupby('patient')['time'].count())
    df['user_max'] = df.patient.map(df.groupby('patient')['time'].max())
    df['user_min'] = df.patient.map(df.groupby('patient')['time'].min())
    df[['time', 'user_count', 'user_max', 'user_min']] /= 1420.0
    return df


def main():

    # fix seed for train reproduction
    seed_everything(args.SEED)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("\n device", device)

    train_df = pd.read_csv('/DATA/trainset-for_user.csv', header=None)
    train_df = train_df.dropna().reset_index(drop=True)
    test_df = pd.read_csv('/DATA/testset-for_user.csv', header=None)

    train_df = create_str_feature(train_df)
    test_df = create_str_feature(test_df)
    
    train_df['patient_label'] = train_df['patient'] + '_' + train_df['label']
    train_df['count'] = train_df['patient_label'].map(train_df['patient_label'].value_counts())

    train_df['image_path'] = [os.path.join('/DATA', train_df['patient'][i], train_df['image'][i]) for i in range(train_df.shape[0])]
    labels = train_df['label'].map({'Wake':0, 'N1':1, 'N2':2, 'N3':3, 'REM':4}).values
    str_train_df = train_df[['time', 'user_count', 'user_max', 'user_min']].values
    str_test_df = test_df[['time', 'user_count', 'user_max', 'user_min']].values

    skf_labels = train_df['patient'] + '_' + train_df['label']

    unique_idx = train_df[train_df['count']==1].index
    non_unique_idx = train_df[train_df['count']>1].index
    trn_idx, val_idx, trn_labels, val_labels = train_test_split(non_unique_idx, labels[non_unique_idx],
                                                                test_size=0.05,
                                                                random_state=0,
                                                                shuffle=True,
                                                                stratify=skf_labels[non_unique_idx])

    # valid set define
    trn_image_paths = train_df.loc[trn_idx, 'image_path'].values
    val_image_paths = train_df.loc[val_idx, 'image_path'].values

    # struture data define
    trn_str_data = str_train_df[trn_idx, :]
    val_str_data = str_train_df[val_idx, :]

    valid_transforms = create_val_transforms(args, args.input_size)
    if args.DEBUG:
        valid_dataset = SleepDataset(args, val_image_paths[:100], val_str_data, val_labels[:100], valid_transforms, is_test=False)
    else:
        valid_dataset = SleepDataset(args, val_image_paths, val_str_data, val_labels, valid_transforms, is_test=False)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)

    trn_skf_labels = (train_df.loc[trn_idx, 'patient'] + train_df.loc[trn_idx, 'label']).values

    if args.DEBUG:
        print('\n#################################### DEBUG MODE')
    else:
        print('\n################################### MAIN MODE')
        print(trn_image_paths.shape, trn_labels.shape, trn_skf_labels.shape)

    # train set define
    train_dataset_dict = {}
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.SEED)
    nsplits = [val_idx for _, val_idx in skf.split(trn_image_paths, trn_skf_labels)]

    for idx, val_idx in enumerate(nsplits): # trn_skf_labels

        sub_img_paths = np.array(trn_image_paths)[val_idx]
        sub_labels = np.array(trn_labels)[val_idx]
        sub_meta = np.array(trn_str_data)[val_idx]
        
        if args.DEBUG:
            sub_img_paths = sub_img_paths[:200]
            sub_labels = sub_labels[:200]
            sub_meta = sub_meta[:200]

        if idx==1 or idx==6:
            sub_img_paths = np.concatenate([sub_img_paths, train_df.loc[unique_idx, 'image_path'].values])
            sub_labels = np.concatenate([sub_labels, labels[unique_idx]])
            sub_meta = np.concatenate([sub_meta, str_train_df[unique_idx]])

        train_transforms = create_train_transforms(args, args.input_size)
        #train_dataset = SleepDataset(args, sub_img_paths, sub_labels, train_transforms, use_masking=True, is_test=False)
        train_dataset_dict[idx] = [args, sub_img_paths, sub_meta, sub_labels, train_transforms]


    # define model
    model = build_model(args, device)

    # optimizer, scheduler definition
    optimizer = build_optimizer(args, model)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 9)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=1, after_scheduler=scheduler_cosine)

    if args.label_smoothing:
        criterion = LabelSmoothingLoss(classes=args.num_classes, smoothing=args.label_smoothing_ratio)
    else:
        criterion = nn.CrossEntropyLoss()

    trn_cfg = {'train_datasets':train_dataset_dict,
                    'valid_loader':valid_loader,
                    'model':model,
                    'criterion':criterion,
                    'optimizer':optimizer,
                    'scheduler':scheduler,
                    'device':device,
                    'fold_num':0,
                    }

    train(args, trn_cfg)


if __name__ == '__main__':
    print(args)
    main()

