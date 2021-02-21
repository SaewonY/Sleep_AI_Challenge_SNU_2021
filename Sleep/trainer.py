from conf import *
from dataloader import *

import gc
import os
import time
import shutil
import numpy as np
from tqdm import tqdm
from glob import glob
from datetime import datetime, timedelta
from sklearn.metrics import f1_score, accuracy_score
import torch
import torch.nn as nn


def train(args, trn_cfg):

    train_datasets = trn_cfg['train_datasets']
    valid_loader = trn_cfg['valid_loader']
    model = trn_cfg['model']
    criterion = trn_cfg['criterion']
    optimizer = trn_cfg['optimizer']
    scheduler = trn_cfg['scheduler']
    device = trn_cfg['device']
    fold_num = trn_cfg['fold_num']

    ### AMP
    scaler = torch.cuda.amp.GradScaler()

    best_epoch = 0
    best_val_score = 0.0

    ########################## 시작하면 폴더생성
    # ctime = time.ctime().replace(' ', '_')[:-4]
    # ctime = (datetime.today() + timedelta(hours=9)).strftime("%Y-%m-%d_%H:%M:%S")
    save_path = f'/USER/INFERENCE/{args.model}_{args.noise_p}/'
    os.makedirs(save_path, exist_ok=True)
    # config file .py로 떨구기
    #default_config_path = f'{args.model}_{args.noise_p}.py'
    #fname = 'experiment.py' # new_fname to choose
    #shutil.copy2(default_config_path, save_path + fname)
    #print("Copying config file From {} To {}".format(default_config_path, save_path+fname))

    # Train the model
    for epoch in range(args.epochs):

        start_time = time.time()
        td_list = train_datasets[epoch%len(train_datasets.keys())]
        train_dataset = SleepDataset(td_list[0], td_list[1], td_list[2], td_list[3], td_list[4], use_masking=True, is_test=False)
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True, drop_last=True)

        trn_loss = train_one_epoch(args, model, criterion, train_loader, optimizer, scheduler, device, scaler)
        val_loss, val_acc, val_score = validation(args, trn_cfg, model, criterion, valid_loader, device)

        elapsed = time.time() - start_time

        lr = [_['lr'] for _ in optimizer.param_groups]

        content = f'Fold {fold_num}, Epoch {epoch}/{args.epochs}, lr: {lr[0]:.7f}, train loss: {trn_loss:.5f}, valid loss: {val_loss:.5f}, val_acc: {val_acc:.4f}, val_f1: {val_score:.4f}, time: {elapsed:.0f}'
        with open(save_path + f'log_{fold_num}.txt', 'a') as appender:
            appender.write(content + '\n')

        # save model weight
        if val_score > best_val_score:
            best_val_score = val_score
            model_save_path = save_path + 'best_score_fold' + str(fold_num) + '_{}.pth'.format(str(epoch).zfill(3))#f'_{epoch}.pth'
            torch.save(model.state_dict(), model_save_path)

        if args.scheduler == 'Plateau':
            scheduler.step(val_score)
        else:
            scheduler.step()

        del train_loader; gc.collect()
        if epoch==10:
            break


def train_one_epoch(args, model, criterion, train_loader, optimizer, scheduler, device, scaler):

    model.train()
    trn_loss = 0.0
    optimizer.zero_grad()

    bar = tqdm(train_loader)
    for data, labels in bar:
        labels = labels.long()
        
        if args.use_meta:
            images = data['image']
            meta = data['meta']
            if device:
                images = images.to(device)
                meta = meta.to(device)
                labels = labels.to(device)
        else:
            images = data['image']
            if device:
                images = images.to(device)
                labels = labels.to(device)

        optimizer.zero_grad()
        if args.fp16:
            if args.use_meta:
                with torch.cuda.amp.autocast():
                    outputs = model(images, meta)
                    loss = criterion(outputs, labels)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            else:
                 with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
  
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
        else:
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
#           optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        trn_loss += loss.item()

        bar.set_description('loss : % 5f' % (loss.item()))
    epoch_train_loss = trn_loss / len(train_loader)

    return epoch_train_loss


def validation(args, trn_cfg, model, criterion, valid_loader, device):

    model.eval()
    val_loss = 0.0
    total_labels = []
    total_outputs = []

    bar = tqdm(valid_loader)
    with torch.no_grad():
        for data, labels in bar:
            labels = labels.long()
            total_labels.append(labels)
            
            if args.use_meta:
                images = data['image']
                meta = data['meta']
                if device:
                    images = images.to(device)
                    meta = meta.to(device)
                    labels = labels.to(device)
                outputs = model(images, meta)
            else: 
                images = data['image']
                if device:
                    images = images.to(device)
                    labels = labels.to(device)
                outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += loss.item()
            total_outputs.append(torch.sigmoid(outputs).cpu().detach().numpy())

            bar.set_description('loss : %.5f' % (loss.item()))

    epoch_val_loss = val_loss / len(valid_loader)

    total_labels = np.concatenate(total_labels).tolist()
    total_outputs = np.concatenate(total_outputs).tolist()
    total_outputs = np.argmax(total_outputs, 1)

    acc = accuracy_score(total_labels, total_outputs)
    metrics = f1_score(total_labels, total_outputs, average='macro')

    return epoch_val_loss, acc, metrics

