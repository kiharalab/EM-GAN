import torch
from torch import nn
from torch.utils.data.dataset import Dataset
from torch.autograd.variable import Variable
from torchvision import transforms
from skimage.transform import resize
from PIL import Image
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

def train_hr_transform():
    return transforms.Compose([
        # transforms.RandomCrop(crop_size),
        transforms.ToTensor(),
])
def train_lr_transform(im_size, upscale_factor):

    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(im_size // upscale_factor, interpolation=Image.BICUBIC),
        transforms.ToTensor()
])

train_set=['6123', '6351', '6481', '6535', '6551', '6526', '6337', '3056', '3179', '3180', '3238', '3337', '3331', '8013', '9537', '9566', '6656', '6668', '6618', '8267', '8289', '4037', '4038', '4040', '4054', '4062', '4071', '4074', '4076', '4077', '4079', '4080', '4083', '4088', '4098', '4138', '3447', '3448', '3524', '3545', '3583', '3601', '3602', '3605', '3654', '3696', '8640', '3727', '3773', '3802', '3776', '8331', '8354', '8398', '8399', '8436', '8454', '8467', '8479', '8481', '8540', '8584', '8585', '8586', '8595', '8608', '8621', '8658', '8702', '8717', '8744', '8745', '8746', '8751', '8794', '8795', '8823', '8840', '6675', '6679', '6700', '6710', '6732', '6733', '6734', '6774']
test_set=["3388","8750","6479","8511","6714","8409","2788","3790","8728","3672","2484","6677","5779","8643","8644","8188","8242","8015","8778","5623","4128","8001","3531","8624","6711","4061","6344","3366","6441","8004","8148","8641","3378","6489","4075","8581","8771","4078","3662","3663"]

class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, im_size, upscale_factor=2, downsample=False,mode=1):
        super(TrainDatasetFromFolder, self).__init__()
        self.im_size=im_size
        self.upscale_factor=upscale_factor
        self.downsample=downsample
        if mode==1  :
            dlist = train_set
        else:
            dlist = test_set
        if self.downsample:
            self.hr_imageFileNames = [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir) if "_LR" in x and x.strip().split("_")[0] in dlist]
            self.lr_transform = train_lr_transform(im_size, upscale_factor)
            self.hr_transform = train_hr_transform()
        else:
            self.lr_imageFileNames = [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir) if "_LR" in x and x.strip().split("_")[0] in dlist]
            self.hr_imageFileNames = [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir) if "_HR" in x and x.strip().split("_")[0] in dlist]
            self.lr_imageFileNames.sort()
            self.hr_imageFileNames.sort()

        print(len(self.hr_imageFileNames))
        print(len(self.lr_imageFileNames))
        print(self.hr_imageFileNames[50001:50010])
        print(self.lr_imageFileNames[50001:50010])

    def __getitem__(self,index):
        hr_filename = self.hr_imageFileNames[index]
        hr_rowdata=np.load(hr_filename)
        
        hr_image = vectors_to_images(hr_rowdata)
        # hr_image = self.hr_transform(hr_image[0])
        # print(hr_image[0])
        if self.downsample:
            lr_image = resize(hr_image[0],(self.im_size // self.upscale_factor,self.im_size // self.upscale_factor,self.im_size // self.upscale_factor),order=3,preserve_range=True)
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
