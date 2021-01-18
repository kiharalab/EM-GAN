import torch
from torch import nn, optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd.variable import Variable
import os
import numpy as np
import argparse

from model import Generator, Discriminator
import utils
from utils import TrainDatasetFromFolder as TDF

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--G_path", help="Generator mdoel path")
	parser.add_argument("--D_path", help="Discriminator mdoel path")
	parser.add_argument("--dir_path", help="path to load LR images and store SR images")
	parser.add_argument("--batch_size", type=int, help="Batch size")
	parser.add_argument("--res_blocks", type=int, help="No. of resnet blocks")
	parser.add_argument("--in_channels", type=int, help="No. of input channels")
	parser.add_argument("--train", type=int, help="Train - 1 or Test - 0")
	parser.add_argument("--downsample", nargs='?', const=True, default=False, help="Downsampling GAN")

	args = parser.parse_args()


	pathG=args.G_path
	pathD=args.D_path
	srDir=args.dir_path

	test_batch_size = args.batch_size
	shuffle_dataset = True
	random_seed= 42

	root=srDir
	dataset = TDF(root,25,2,args.downsample,args.train) 
	dataset_size = len(dataset)
	print(dataset_size)
	indices = list(range(dataset_size))
	if shuffle_dataset :
	    np.random.seed(random_seed)
	    np.random.shuffle(indices)

	indices_test=indices

	print(len(indices_test))
	# Creating PT data samplers and loaders:
	test_sampler = SubsetRandomSampler(indices_test)
	test_loader = torch.utils.data.DataLoader(dataset, batch_size=test_batch_size, 
	                                           sampler=test_sampler)


	# Num batches
	num_batches=len(test_loader)
	print(num_batches)

	G = Generator(args.in_channels,2,args.res_blocks, args.downsample)
	G.load_state_dict(torch.load(pathG))
	G.eval()
	D = Discriminator(args.in_channels)
	D.load_state_dict(torch.load(pathD))
	D.eval()

	G.cuda()
	D.cuda()

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

			utils.write_voxels(args.batch_size, srDir, sr_test, index, args.downsample, "test",coords,filName)
			if (index) % 50 == 0: 
				print(index)

		print(torch.cuda.memory_allocated())

if __name__=="__main__":
	main()

