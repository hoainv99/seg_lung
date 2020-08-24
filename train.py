import torch
from torch.utils.data import DataLoader
from torch.optim import  lr_scheduler,Adam
import torchvision 
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import time
from torchvision import  transforms
from loss import dice_loss
from dataset import MyDataset
from utils import get_data_dict,compute_iou
from sklearn.model_selection import train_test_split
from model import ResNetUNet
import torch.nn.functional as F
class Trainer():
    def __init__(self, config, model, optim,pretrained=False):

        self.config = config

        self.device = config['device']
        self.num_iters = config['trainer']['iters']

        self.image_path = config['dataset']['image_path']
        self.label_path = config['dataset']['label_path']

        self.batch_size_train = config['trainer']['batch_size']
        self.print_every = config['trainer']['print_every']
        self.valid_every = config['trainer']['valid_every']

        self.batch_size_val = config['val']['batch_size']
        self.batch_size_test = config['test']['batch_size']

        self.checkpoint = config['val']['checkpoint']
        self.model = model
        print(self.model)
        if pretrained:
            self.load_checkpoint(self.checkpoint)

        self.iter = 0

        self.optimizer = optim
        self.exp_lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1) 
        
        trans = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.RandomErasing(),
        ])
        #get data dict 
        self.data_dict = get_data_dict(self.image_path,self.label_path)
        #split data to train val test
        print(len(self.data_dict))
        self.train_data_dict,self.val_test = train_test_split(self.data_dict, random_state=42, test_size=0.2)
        self.val_data_dict,self.test_data_dict = train_test_split(self.val_test,random_state=42,test_size=0.5)
        #init data loader
        self.train_data_loader = DataLoader(MyDataset(self.train_data_dict,trans),batch_size=self.batch_size_train, shuffle=True,num_workers=4)
        self.val_data_loader = DataLoader(MyDataset(self.val_data_dict),batch_size=self.batch_size_val, shuffle=False, num_workers=4)
        self.test_data_dict = DataLoader(MyDataset(self.test_data_dict),shuffle=False,batch_size=self.batch_size_test,num_workers=4)

        self.train_losses = []


    def train(self):
        total_loss = 0
        
        total_loader_time = 0
        total_gpu_time = 0

        data_iter = iter(self.train_data_loader)
        for i in range(self.num_iters):
            self.iter += 1

            start = time.time()

            try:
                image,label = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_data_loader)
                image,label = next(data_iter)

            total_loader_time += time.time() - start

            start = time.time()
            loss = self.step(image,label)
            total_gpu_time += time.time() - start

            total_loss += loss
            self.train_losses.append((self.iter, loss))

            if self.iter % self.print_every == 0:
                info = 'iter: {:06d} - train loss: {:.3f} - lr: {:.2e} - load time: {:.2f} - gpu time: {:.2f}'.format(self.iter, 
                        total_loss/self.print_every, self.optimizer.lr, 
                        total_loader_time, total_gpu_time)

                total_loss = 0
                total_loader_time = 0
                total_gpu_time = 0
                print(info) 

            if self.valid_annotation and self.iter % self.valid_every == 0:
                val_loss,iou = self.validate()

                info = 'iter: {:06d} - valid loss: {:.3f} - iou: {:.4f} '.format(self.iter, val_loss, iou)
                print(info)

                self.save_checkpoint(self.checkpoint)
            
    def validate(self):
        self.model.eval()

        total_loss = []
        pred_label = []
        target_label = []
        with torch.no_grad():
            for step, image,label in enumerate(self.val_data_loader):
                batch = self.batch_to_device(image,label)

                outputs = self.model(batch['img'])
                
                loss = self.criterion(outputs, batch['label'])

                total_loss.append(loss.item())
                
                pred_label.append(outputs)
                target_label.append(label)
                del outputs
                del loss

        total_loss = np.mean(total_loss)
        self.model.train()
        
        return total_loss,compute_iou(pred_label,target_label)
    
    def predict(self, sample=None):
        pred_sents = []
        actual_sents = []
        img_files = []

        for batch in  self.valid_gen:
            batch = self.batch_to_device(batch)

            if self.beamsearch:
                translated_sentence = batch_translate_beam_search(batch['img'], self.model)
            else:
                translated_sentence = translate(batch['img'], self.model)

            pred_sent = self.vocab.batch_decode(translated_sentence.tolist())
            actual_sent = self.vocab.batch_decode(batch['tgt_input'].T.tolist())

            img_files.extend(batch['filenames'])

            pred_sents.extend(pred_sent)
            actual_sents.extend(actual_sent)
            
            if sample != None and len(pred_sents) > sample:
                break

        return pred_sents, actual_sents, img_files

    
    def visualize_prediction(self, sample=16):
        
        pred_sents, actual_sents, img_files = self.predict(sample)
        img_files = img_files[:sample]
        
        for vis_idx in range(0, len(img_files)):
            img_path = img_files[vis_idx]
            pred_sent = pred_sents[vis_idx]
            actual_sent = actual_sents[vis_idx]

            img = Image.open(open(img_path, 'rb'))
            plt.figure()
            plt.imshow(img)
            plt.title('pred: {} - actual: {}'.format(pred_sent, actual_sent), loc='left')
            plt.axis('off')

        plt.show()
    
    # def visualize_dataset(self, sample=5):
    #     n = 0
    #     image,label = next(iter(self.train_data_loader)
    #     for i in range(5):
    #         img = image[i].numpy().transpose(1,2,0)
    #         mask = label[i].numpy()
    #         lb = mask.unsqueeze(-1).repeat(1,1,3)
    #         lb = lb.masked_fill(mask.unsqueeze(-1).type(torch.BoolTensor), value = 255.)
            
    #         plt.figure()
    #         plt.title('sent: {}'.format(sent), loc='center')
    #         plt.imshow(img)
    #         plt.imshow(lb)
    #         plt.axis('off')
    #     plt.show()


    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
    
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.model.load_state_dict(checkpoint['state_dict'])
        self.iter = checkpoint['iter']

        self.train_losses = checkpoint['train_losses']

    def save_checkpoint(self, filename):
        state = {'iter':self.iter, 'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(), 'train_losses': self.train_losses}
        
        path, _ = os.path.split(filename)
        os.makedirs(path, exist_ok=True)

        torch.save(state, filename)

    def batch_to_device(self, image,label):
        img = image.to(self.device, non_blocking=True)
        label = label.to(self.device, non_blocking=True)

        batch = {
                'img': img, 'label':label, 
                }

        return batch


    def step(self, image,label):
        self.model.train()

        batch = self.batch_to_device(image,label)
        outputs = self.model(batch['img'])
        print(outputs.shape)
        loss = self.calc_loss(outputs, batch['label'])

        self.optimizer.zero_grad()

        loss.backward()
        
        self.optimizer.step()

        loss_item = loss.item()

        return loss_item
