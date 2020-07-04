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
	parser.add_argument("--G_path", help="Generator mdoel path")
	parser.add_argument("--D_path", help="Discriminator mdoel path")
	parser.add_argument("--vis_path", help="Store SR images path")
	parser.add_argument("--batch_size", type=int, help="Batch size")
	parser.add_argument("--res_blocks", type=int, help="No. of resnet blocks")
	parser.add_argument("--in_channels", type=int, help="No. of input channels")
	parser.add_argument("--dropout", type=float, help="Dropout")
	parser.add_argument("--train", type=int, help="Train - 1 or Test - 0")
	parser.add_argument("--downsample", nargs='?', const=True, default=False, help="Downsampling GAN")
	parser.add_argument("--loss", help="Loss function - mse/bce/comb")
	parser.add_argument("--norm", help="BN or IN for Generator")

	args = parser.parse_args()
	print(args.batch_size)
	print(args.res_blocks)
	print(args.in_channels)


	pathG=args.G_path
	pathD=args.D_path
	srDir=args.vis_path

	test_batch_size = args.batch_size
	shuffle_dataset = True
	random_seed= 42

	# Creating data indices for training and validation splits:
	# dataset_size = len(data_loader)*batch_size
	# root='/net/kihara/scratch/smaddhur/SuperReso/dataset_clean'
	root='/net/kihara-fast-scratch/smaddhur/SuperReso/EM_data/dataset_new_2_6'
	dataset = TDF(root,25,2,args.downsample,args.train) 
	dataset_size = len(dataset)
	print(dataset_size)
	indices = list(range(dataset_size))
	if shuffle_dataset :
	    np.random.seed(random_seed)
	    np.random.shuffle(indices)

	# indices_test=indices[900000:900160]
	indices_test=indices

	print(len(indices_test))
	# Creating PT data samplers and loaders:
	test_sampler = SubsetRandomSampler(indices_test)
	test_loader = torch.utils.data.DataLoader(dataset, batch_size=test_batch_size, 
	                                           sampler=test_sampler)


	# Num batches
	num_batches=len(test_loader)
	print(num_batches)

	G = Generator(args.in_channels,2,args.res_blocks, args.downsample, args.dropout, args.norm)
	G.load_state_dict(torch.load(pathG))
	G.eval()
	D = Discriminator(args.in_channels, args.dropout)
	D.load_state_dict(torch.load(pathD))
	D.eval()

	G.cuda()
	D.cuda()

	G_criterionLoss = GeneratorLoss(args.loss).cuda()

	coords=["0","0","0"]
	with torch.no_grad():
		for index,(lr, hr, filName) in enumerate(test_loader):

			lr=lr.float()
			val_z=Variable(lr)
			val_z=val_z.cuda()
			sr_test = G(val_z)
			val_target=Variable(hr)
			val_target=val_target.cuda()
			hr_test=D(val_target).mean()
			hr_fake=D(sr_test).mean()

			G_loss_test=G_criterionLoss(hr_fake, sr_test, val_target)
			D_loss_test=1 - hr_test + hr_fake
			utils.write_voxels(args.batch_size, srDir, sr_test, index, args.downsample, "test",coords,filName)
			if (index) % 50 == 0: 
				print(index)

		print(torch.cuda.memory_allocated())
		print('TEST - ')
		print("D_loss:%d,G_loss:%",(D_loss_test, G_loss_test))

		del G_loss_test
		del D_loss_test



if __name__=="__main__":
	main()

