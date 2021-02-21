#!/usr/bin/env python
# -*- coding: utf-8 -*-

from conf import *

import os
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from dataloader import *
from transforms import *
from models import *


def create_str_feature(df):
    df = df.rename(columns={0:'patient', 1:'image'})
    df['time'] = df['image'].apply(lambda x : int(x.split('_')[-1][:-4]))
    df['user_count'] = df.patient.map(df.groupby('patient')['time'].count())
    df['user_max'] = df.patient.map(df.groupby('patient')['time'].max())
    df['user_min'] = df.patient.map(df.groupby('patient')['time'].min())
    df[['time', 'user_count', 'user_max', 'user_min']] /= 1420.0
    return df


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device", device)
test_df = pd.read_csv('/DATA/testset-for_user.csv', header=None)
test_df = create_str_feature(test_df)
test_image_paths = [os.path.join('/DATA', test_df['patient'][i], test_df['image'][i]) for i in range(test_df.shape[0])]
str_test_df = test_df[['time', 'user_count', 'user_max', 'user_min']].values

if args.DEBUG:
    test_image_paths = test_image_paths[:500]

print(test_df.shape, len(test_image_paths))

bs = args.infer_batch_size
test_transforms = create_val_transforms(args, args.input_size)

test_dataset = SleepDataset(args, 
                            image_paths=test_image_paths, 
                            meta_data=str_test_df,
                            labels=None,
                            transforms=test_transforms, 
                            is_test=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=bs, num_workers=8, shuffle=False, pin_memory=True)


def inference(model, test_loader, device):

    test_preds = np.zeros((len(test_loader.dataset), 5))
    bar = tqdm(test_loader)
    with torch.no_grad():
        for i, data in enumerate(bar):
            images = data['image']
            images = images.to(device)
            if args.use_meta:
                meta = data['meta']
                meta = meta.to(device)
                outputs = model(images, meta)
            else:
                outputs = model(images)
            test_preds[i*bs:(i+1)*bs, :] = torch.sigmoid(outputs).detach().cpu().numpy()
    return test_preds

model = build_model(args, device) 

model_path = '/USER/INFERENCE'
folder = args.infer_model_path#'2021-02-04_17:26:13_tf_efficientnet_b0_ns'
file_name = args.infer_best_model_name#'best_score_fold0_029.pth'
model.load_state_dict(torch.load(os.path.join(model_path, folder, file_name)))
model.eval()

test_preds = inference(model, test_loader, device)
np.save(os.path.join(model_path, f'test_preds{args.infer_prefix}.npy'), test_preds)

test_preds = np.argmax(test_preds, 1)
result_df = pd.DataFrame(test_preds)
label_dict = {0:'Wake', 1:'N1', 2:'N2', 3:'N3', 4:'REM'}
result_df[0] = result_df[0].map(label_dict)

test_pred_path = "/USER/INFERENCE"
result_df.to_csv(os.path.join(test_pred_path, f'final_result{args.infer_prefix}.csv'), header=None, index=False)

