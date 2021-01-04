import argparse
import os
import numpy as np
from tqdm import tqdm
import yaml
import torch
import math
from addict import Dict

import torch.nn.functional as F
from time import time
from libs.optimizers import get_optimizer
from libs.models import get_network
from libs.loss import get_lossfunction
from libs.datasets.base import myDataset
from libs.utils import saver, metric, LR_Scheduler

class Trainer(object):
    def __init__(self, config_path):
        config = Dict(yaml.load(open(config_path,'r'), Loader=yaml.FullLoader))
        self.args = config 

        ## Define Saver
        self.saver = saver.Saver(self.args)
        self.saver.save_experiment_config()

        self.dim = self.args.dataset.dim
        self.channel = self.args.dataset.channel
        self.n_classes = self.args.dataset.n_classes

        
        ## Define Evaluator
        self.evaluator = metric.Evaluator(self.n_classes)
        ## define dataloader of train and validation
        train_dataset = myDataset(
            root = self.args.dataset.root,
            split = self.args.dataset.split.labeled,
            ignore_label = self.args.dataset.ignore_label,
            augment = True,
            crop_size = self.args.image.size.train,
        )
        self.train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,batch_size=self.args.solver.batch_size.train,
            num_workers=self.args.dataloader.num_workers,shuffle=True,
        )
        
        val_dataset = myDataset(
            root = self.args.dataset.root,
            split = self.args.dataset.split.valid,
            ignore_label = self.args.dataset.ignore_label,
            augment = True,
            crop_size = self.args.image.size.test,
        )
        self.val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=self.args.solver.batch_size.test,
            num_workers=0,#self.args.DATALOADER.NUM_WORKERS,
            shuffle=False,
        )

        # Define network
        network_cls = get_network(config)
        network_param = {k: v for k, v in config['network'].items() if k != 'name'}
        model_Unet = network_cls(**network_param)
        self.model_Unet = model_Unet
        if self.args.exp.cuda:
            self.model_Unet = torch.nn.DataParallel(self.model_Unet).cuda()
        
        # Define Optimizer
        optimizer_cls = get_optimizer(config)
        optimizer_params = {k: v for k, v in config['solver']['optimizer'].items() if k != 'name'}
        optimizer_Unet = optimizer_cls(self.model_Unet.parameters(), **optimizer_params)
        print("Using optimizer{}".format(optimizer_Unet))
        self.optimizer_Unet = optimizer_Unet
        # Define Criterion
        loss_cls = get_lossfunction(config)
        loss_params = {k: v for k, v in config['solver']['loss_function'].items() if k != 'loss_type'}
        self.criterion = loss_cls(**loss_params)
        
        # Define lr scheduler
        self.scheduler = LR_Scheduler(self.args.solver.lr_scheduler, self.args.solver.optimizer.lr,
                                            self.args.solver.epoch_max, len(self.train_loader))

        # Resuming checkpoint
        self.best_pred = 0.0
        if self.args.init_model != 'None':
            if not os.path.isfile(self.args.init_model):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(self.args.init_model))
            checkpoint = torch.load(self.args.init_model)
            self.args.solver.epoch_start = checkpoint['epoch']
            if self.args.exp.cuda:
                self.model_Unet.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model_Unet.load_state_dict(checkpoint['state_dict'])
            if not self.args.ft:
                self.optimizer_Unet.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(self.args.init_model, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if self.args.ft:
            self.args.solver.epoch_start = 0

    def training(self, epoch):
        torch.autograd.set_detect_anomaly(True)
        train_loss = 0.0
        self.model_Unet.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            ids, image, target = sample
            if self.args.exp.cuda:
                image, target = image.cuda(), target.cuda()
            for p in self.model_Unet.parameters():
                p.requires_grad = True 
            # self.scheduler(self.optimizer_Unet, i, epoch, self.best_pred)
            self.optimizer_Unet.zero_grad()
            output = self.model_Unet(image)
            loss = self.criterion(output, target.long())
            loss.backward(retain_graph=True)
            train_loss = loss.detach().item() + train_loss
            self.optimizer_Unet.step()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
           
                
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.solver.batch_size.train + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model_Unet.module.state_dict(),
                'optimizer': self.optimizer_Unet.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


    def validation(self, epoch):
        def split_valdata(image_shape,crop_size):
            num_and_pad = []
            for i in range(len(image_shape)):
                num = math.ceil(image_shape[i]/crop_size[i])
                pad = crop_size[i]-image_shape[i]/num
                num_and_pad.append(num)
                num_and_pad.append(pad)
            return num_and_pad

        self.model_Unet.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            ids, image, target = sample
            if self.args.exp.cuda:
                image = image.cuda()
                target = target.cuda()
            with torch.no_grad():
                B,C,D,H,W = image.shape
                crop_d,crop_h,crop_w = self.args.image.size.train
                num_d,pad_d,num_h,pad_h,num_w,pad_w = split_valdata(image.shape[2:],self.args.image.size.train)
                output = torch.zeros((B,self.args.dataset.n_classes,D,H,W)).cuda()
                output_weight = torch.zeros(output.shape).cuda()
                for index_d in range(num_d):
                    for index_h in range(num_h):
                        for index_w in range(num_w):
                            start_d = index_d*crop_d-int(pad_d*index_d)
                            end_d = index_d*crop_d-int(pad_d*index_d)+crop_d
                            start_h = index_h*crop_h-int(pad_h*index_h)
                            end_h = index_h*crop_h-int(pad_h*index_h)+crop_h
                            start_w = index_w*crop_w-int(pad_w*index_w)
                            end_w = index_w*crop_w-int(pad_w*index_w)+crop_w
                            image_crop = image[:,:,start_d:end_d,start_h:end_h,start_w:end_w]
                            # output[:,:,start_d:end_d,start_h:end_h,start_w:end_w] = torch.unsqueeze(F.softmax(self.model_Unet(image_crop),dim=1),dim=1)
                            output[:,:,start_d:end_d,start_h:end_h,start_w:end_w] = self.model_Unet(image_crop)
                            output_weight[:,:,start_d:end_d,start_h:end_h,start_w:end_w] += 1
                output = output/output_weight
            loss = self.criterion(output, target.long())
            # import pdb; pdb.set_trace()
            test_loss = loss.detach().item()+test_loss
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.solver.batch_size.train + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model_Unet.module.state_dict(),
                'optimizer': self.optimizer_Unet.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

    

def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--configfile', type=str, default='configs/Gan.yaml',
                        help='config file path')

    args = parser.parse_args()
    print(args)
    torch.backends.cudnn.benchmark = True
    trainer = Trainer(args.configfile)
    print('Starting Epoch:', trainer.args.solver.epoch_start)
    print('Total Epoches:', trainer.args.solver.epoch_max)
    for epoch in range(trainer.args.solver.epoch_start, trainer.args.solver.epoch_max):
        trainer.training(epoch)
        if epoch % trainer.args.solver.epoch_save == (trainer.args.solver.epoch_save - 1):
            trainer.validation(epoch)

    trainer.writer.close()

if __name__ == "__main__":
   main()
