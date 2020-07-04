import torch
from torch import nn, optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd.variable import Variable
import os
from tqdm import tqdm
import numpy as np
import argparse
import calendar, time

from logger import Logger
from model import Generator, Discriminator
import utils
from utils import TrainDatasetFromFolder as TDF
from loss import GeneratorLoss


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--batch_size", type=int, help="Batch size")
	parser.add_argument("--res_blocks", type=int, help="No. of resnet blocks")
	parser.add_argument("--in_channels", type=int, help="No. of input channels")
	parser.add_argument("--w_decay", help="Weight decay")
	parser.add_argument("--lr", type=float, help="Learning Rate")
	parser.add_argument("--dropout", type=float, help="Dropout")
	parser.add_argument("--downsample", nargs='?', const=True, default=False, help="Downsampling GAN")
	parser.add_argument("--train", type=int, help="Train - 1 or Test - 0")
	parser.add_argument("--loss", default="mse",  help="Loss function - mse/bce/comb")
	parser.add_argument("--vis_path", help="Store SR images path")
	parser.add_argument("--save_state", nargs='?', const=True, default=False, help="Specify this to NOT save state and loss")
	parser.add_argument("--norm", help="BN or IN for Generator")

	args = parser.parse_args()
	print(args.batch_size)
	print(args.res_blocks)
	print(args.in_channels)
	print(args.w_decay)


	# root='/net/kihara/scratch/smaddhur/SuperReso/dataset_clean'
	root = '/net/kihara-fast-scratch/smaddhur/SuperReso/EM_data/dataset_new_2_6'

	dataset = TDF(root,25,2,args.downsample,args.train)

	# Load data
	# Create loader with data
	batch_size = args.batch_size
	val_batch_size = args.batch_size
	validation_per = 0.1
	shuffle_dataset = True
	random_seed= 42

	# data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

	# Creating data indices for training and validation splits:
	# dataset_size = len(data_loader)*batch_size
	dataset_size = len(dataset)
	print(dataset_size)
	indices = list(range(dataset_size))
	if shuffle_dataset :
	    np.random.seed(random_seed)
	    np.random.shuffle(indices)

	# indices_small=indices[:220000]
	indices_small=indices
	#split = int(np.floor(validation_per * dataset_size))
	split = 10000

	# print(indices_small[0:50])
	train_indices, val_indices = indices_small[split:], indices_small[:split]
	print(len(train_indices))
	# Creating PT data samplers and loaders:
	train_sampler = SubsetRandomSampler(train_indices)
	valid_sampler = SubsetRandomSampler(val_indices)

	train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
	                                           sampler=train_sampler)
	validation_loader = torch.utils.data.DataLoader(dataset, batch_size=val_batch_size,
	                                                sampler=valid_sampler)


	# Num batches
	num_batches=len(train_loader)
	print(len(train_loader))
	print(len(validation_loader))


	res_blocks=args.res_blocks
	in_channels=args.in_channels
	w_decay=float(args.w_decay)
	lr = args.lr
	train_epoch = 50
	logger = Logger(model_name='SuperEMGAN', data_name='EM')
	G = Generator(in_channels,2,res_blocks, args.downsample, args.dropout, args.norm)
	D = Discriminator(in_channels,args.dropout)

	G.cuda()
	D.cuda()

	G_criterionLoss = GeneratorLoss(args.loss).cuda()

	# G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999),weight_decay=w_decay)
	# D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999),weight_decay=w_decay)

	G_optimizer = optim.Adam(G.parameters(), lr=lr)
	D_optimizer = optim.Adam(D.parameters(), lr=lr)

	ts = calendar.timegm(time.gmtime())

	evalDir="/net/kihara/home/smaddhur/tensorFlow/SuperReso/SuperEM_eval/run_final_new_data_11_22/"
	out_dir="/net/kihara/home/smaddhur/tensorFlow/SuperReso/data/SuperEMGAN/run_final_new_data_11_22/"
	srDir=args.vis_path
	for epoch in range(train_epoch):
	    tl=tqdm(train_loader)
	    for n_batch,(x_, target, _ ) in enumerate(tl):
	        # train discriminator D and generator G
	        G.train()
	        D.train()
	        D.zero_grad()

	        mini_batch = x_.size()[0]
	        # mini_batch = batch_size
	        y_real_ = torch.ones(mini_batch)
	        y_fake_ = torch.zeros(mini_batch)      
	        
	        real_img=Variable(target)
	        if torch.cuda.is_available():
	            real_img=real_img.cuda()
	        x_=x_.float()
	        fake_z=Variable(x_)
	        if torch.cuda.is_available():
	            fake_z=fake_z.cuda()

	               
	        fake_img=G(fake_z)
	#         print("real D start")
	        realD=D(real_img).mean()
	#         print("Fake starts")
	        fakeD=D(fake_img).mean()
	        D_real_loss = 1 - realD + fakeD
	        D_real_loss.backward(retain_graph=True)
	        D_optimizer.step()

	        G.zero_grad()
	        G_train_loss = G_criterionLoss(fakeD,fake_img,real_img)
	        G_train_loss.backward()
	        G_optimizer.step() 

	        if (n_batch) % 1000 == 0: 
	            print('TRAIN - ')
	            logger.display_status(
	                (epoch+1), train_epoch, n_batch, num_batches,
	                D_real_loss, G_train_loss,0,0) 


	            if not args.save_state:
	                utils.save_loss_state(evalDir, out_dir, ts, G, D, epoch, n_batch, G_train_loss, D_real_loss)
	            else:
	            	utils.write_voxels(batch_size, srDir, x_, fake_img, target, n_batch, args.downsample, "train",["0","0","0"])

	        del D_real_loss
	        del G_train_loss
	        if (n_batch) % 2500 == 0 and n_batch!=0:             
	            G.eval()
	            D.eval()
	            with torch.no_grad():
	                for _,(lr, hr,_) in enumerate(validation_loader):
	                    lr=lr.float()
	                    val_z=Variable(lr)
	                    val_z=val_z.cuda()
	                    sr_test = G(val_z)
	                    val_target=Variable(hr)
	                    val_target=val_target.cuda()
	                    hr_test=D(val_target).mean()
	                    hr_fake=D(sr_test).mean()
	                    
	                    G_loss_valid=G_criterionLoss(hr_fake, sr_test, val_target)
	                    D_loss_valid=1 - hr_test + hr_fake
	                
	                print(torch.cuda.memory_allocated())

	                if not args.save_state:
	                    utils.save_valid_loss(evalDir, ts, epoch, n_batch, G_loss_valid, D_loss_valid)                

	                print('VALIDATION - ')
	                logger.display_status(
	                    (epoch+1), train_epoch, n_batch, num_batches,
	                    D_loss_valid, G_loss_valid, 0, 0) 
	                del G_loss_valid
	                del D_loss_valid

if __name__=="__main__":
	main()

