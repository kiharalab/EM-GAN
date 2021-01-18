import torch
from torch import nn
from torch.utils.data.dataset import Dataset
from torch.autograd.variable import Variable
# from torchvision import transforms
# from skimage.transform import resize
# from PIL import Image
import os
import numpy as np

def images_to_vectors(images):
    return images.view(images.size(0), 15625)

#Change
def vectors_to_images(vectors):
    #print(len(vectors))
    return np.reshape(vectors,(1,25, 25,25))

def noise(size):
    '''
    Generates a 1-d vector of gaussian sampled random values
    '''
    n = Variable(torch.randn(size, 100).view(-1, 100, 1, 1).cuda())
    return n

def ones_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 1)).long()
    return data

def zeros_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 1)).long()
    return data

#Change
def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


def calc_pad(k,d):
    return int((k-1)*d/2)

def update_exp(exp):
    return (exp+1) if (exp<=8) else 8    

def pixel_shuffle(input,up_scale):
    input_size = list(input.size())
    dimensionality = len(input_size) - 2

    input_size[1] //= (upscale_factor ** dimensionality)
    output_size = [dim * upscale_factor for dim in input_size[2:]]

    input_view = input.contiguous().view(
        input_size[0], input_size[1],
        *(([upscale_factor] * dimensionality) + input_size[2:])
    )

    indicies = list(range(2, 2 + 2 * dimensionality))
    indicies = indicies[1::2] + indicies[0::2]

    shuffle_out = input_view.permute(0, 1, *(indicies[::-1])).contiguous()
    return shuffle_out.view(input_size[0], input_size[1], *output_size)

# def train_hr_transform():
#     return transforms.Compose([
#         # transforms.RandomCrop(crop_size),
#         transforms.ToTensor(),
# ])
# def train_lr_transform(im_size, upscale_factor):

#     return transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.Resize(im_size // upscale_factor, interpolation=Image.BICUBIC),
#         transforms.ToTensor()
# ])

class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, im_size, upscale_factor=2, downsample=False,mode=1):
        super(TrainDatasetFromFolder, self).__init__()
        self.im_size=im_size
        self.upscale_factor=upscale_factor
        self.downsample=downsample
        if self.downsample:
            self.hr_imageFileNames = [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir) if "_LR" in x and x.strip().split("_")[0]]
            self.lr_transform = train_lr_transform(im_size, upscale_factor)
            self.hr_transform = train_hr_transform()
        else:
            self.lr_imageFileNames = [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir) if "_LR" in x and x.strip().split("_")[0]]
            self.hr_imageFileNames = [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir) if "_HR" in x and x.strip().split("_")[0]]
            self.lr_imageFileNames.sort()
            self.hr_imageFileNames.sort()


    def __getitem__(self,index):
        hr_filename = self.hr_imageFileNames[index]
        hr_rowdata=np.load(hr_filename)
        
        hr_image = vectors_to_images(hr_rowdata)
        # hr_image = self.hr_transform(hr_image[0])
        # print(hr_image[0])
        if self.downsample:
            # lr_image = resize(hr_image[0],(self.im_size // self.upscale_factor,self.im_size // self.upscale_factor,self.im_size // self.upscale_factor),order=3,preserve_range=True)
            # print(lr_image.shape)
            lr_image = np.array([lr_image])
            # lr_image = self.lr_transform(hr_image[0])
        else:
            lr_filename = self.lr_imageFileNames[index]
            lr_rowdata=np.load(lr_filename)
            lr_image = vectors_to_images(lr_rowdata)

        return lr_image, hr_image, hr_filename

    def __len__(self):
        return len(self.hr_imageFileNames)


def write_voxels(batch_size, srDir, sr, index, downsample, mtype,coords,fn):

    coords_str = " ".join(coords)
    # print(len(sr)!=16)
    for it in range(len(sr)):
        # print(index*batch_size + it)
        # validLRFile = open(srDir+mtype+"LR_"+str(index*batch_size + it)+".situs","w")
        # # if downsample:
        # #     validLRFile.write("1.000000 "+coords_str+" 12 12 12\n\n")
        # # else:
        # #     validLRFile.write("1.000000 "+coords_str+" 25 25 25\n\n")
        # for i in np.array(lr.data.cpu().tolist()[it][0]).flatten():
        #     validLRFile.write(str(i)+" ")
        # validLRFile.close()
        filName = fn[it].strip().split("/")[-1]
        filName = filName.split("_")[0]+"_"+filName.split("_")[2][2:].split(".")[0]
        validSRFile = open(srDir+mtype+"SR_"+filName+".situs","w")
        # validSRFile.write("1.000000 "+coords_str+" 25 25 25\n\n")

        for i in np.array(sr.data.cpu().tolist()[it][0]).flat:
            validSRFile.write(str(i)+" ")
        validSRFile.close()
        


        # validHRFile = open(srDir+mtype+"HR_"+str(index*batch_size + it)+".situs","w")
        # validHRFile.write("1.000000 "+coords_str+" 25 25 25\n\n")
        # for i in np.array(hr.data.cpu().tolist()[it][0]).flatten():
        #     validHRFile.write(str(i)+" ")
        # validHRFile.close()     


def save_loss_state(evalDir, out_dir, ts, G, D, epoch, n_batch, G_train_loss, D_real_loss):
    trainGFile = open(os.path.join(evalDir,"trainG"+"_"+str(ts)),"a")
    trainGFile.write(str((epoch+1)*n_batch)+":"+str(G_train_loss.data.cpu().tolist())+",")
    trainGFile.close()
    trainDFile = open(os.path.join(evalDir,"trainD"+"_"+str(ts)),"a")
    trainDFile.write(str((epoch+1)*n_batch)+":"+str(D_real_loss.data.cpu().tolist())+",")
    trainDFile.close()
    torch.save(G.state_dict(),
           os.path.join(out_dir,'G_epoch'+"_"+str(ts)+'_'+str(epoch+1)))
    torch.save(D.state_dict(),
           os.path.join(out_dir,'D_epoch'+"_"+str(ts)+'_'+str(epoch+1)))

def save_valid_loss(evalDir, ts, epoch, n_batch, G_loss_valid, D_loss_valid):
    validGFile = open(os.path.join(evalDir,"validG"+"_"+str(ts)),"a")
    validGFile.write(str((epoch+1)*n_batch)+":"+str(G_loss_valid.data.cpu().tolist())+",")
    validGFile.close()
    validDFile = open(os.path.join(evalDir,"validD"+"_"+str(ts)),"a")
    validDFile.write(str((epoch+1)*n_batch)+":"+str(D_loss_valid.data.cpu().tolist())+",")
    validDFile.close()    
