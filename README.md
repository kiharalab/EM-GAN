# ContactGAN
ContactGAN takes a predicted protein contact map as input and outputs a new contact map that better captures the residue-residue contacts

Copyright (C) 2020 Sai Raghavendra Maddhuri, Aashish Jain, Yuki Kagaya, Genki Terashi, Daisuke Kihara, and Purdue University.

License: GPL v3 for academic use. (For commercial use, please contact us for different licensing)

Contact: Daisuke Kihara (dkihara@purdue.edu)

Cite : Sai Raghavendra Maddhuri Venkata Subramaniya, Aashish Jain, Yuki Kagaya, Genki Terashi, and Daisuke Kihara, Protein Contact Map De-noising using Generative Adversarial Networks. In Submission. (2020)

## About ContactGAN  
> ContactGAN is a novel contact map denoising and refinement method based on Generative adversarial networks.  
> ContactGAN can be trained and combined with any contact prediction method to improve and correct similar patterns of errors that creep into the method.
> Currently ContactGAN is trained and works with 4 contact prediction methods - CCMPred, DeepContact, DeepCov and trRosetta
![](https://github.com/kiharalab/ContactGAN/blob/master/data/git/fig1.png)   


## Pre-required software
```

Python 3 : https://www.python.org/downloads/  
pytorch : pip/conda install pytorch  
CCMPred : A freely available software. It can be downloaded and installed from here : https://github.com/soedinglab/CCMpred  
DeepContact : A freely available software. It can be downloaded and installed from here : https://github.com/largelymfs/deepcontact   
DeepCov : A freely available software. It can be downloaded and installed from here : https://github.com/psipred/DeepCov  

```
## Instructions  
Generate an input contact map file using a method of your choice from the 4 methods described.  
### ContactGAN Usage  
If you are testing with a single method as 1-channel input, run ContactGAN as follows:  
```
python test/denoising_gan_test.py --input=<INPUT contact prediction directory> --G_res_blocks=3 --D_res_blocks=3 --G_path=model/<1-channel directory>/Generator --D_path=model/<1-channel directory>/Discriminator
  --input               Input Contact Map  Directory  
  --G_res_blocks        Number of ResNet blocks in Generator (Default : 3)
  --D_res_blocks        Number of ResNet blocks in Discriminator (Default : 3)
  --G_path              Specify path of Generator model
  --D_path              Specify path of Discriminator model
  
```
If you are testing with two methods as 2-channel input, run ContactGAN as follows:  
```
python test/denoising_gan2_test.py --input <INPUT contact prediction directory 1> <INPUT contact prediction directory 2> --G_res_blocks=3 --D_res_blocks=3 --G_path=model/<2-channel directory>/Generator --D_path=model/<2-channel directory>/Discriminator
 
```
If you are testing with three methods as 3-channel input, run ContactGAN as follows:  
```
python test/denoising_gan3_test.py --input <INPUT contact prediction directory 1> <INPUT contact prediction directory 2> <INPUT contact prediction directory 3> --G_res_blocks=3 --D_res_blocks=3 --G_path=model/<3-channel directory>/Generator --D_path=model/<3-channel directory>/Discriminator
 
```
### Output interpretation  
Generated output contact map file is the denoised version of the input map.Output file looks exactly same as input file structure-wise.  
An example contact map can be found at the bottom of this page.  
### Visualization    
```
python util/plot_cmap.py --input=<OUTPUT Contact Prediction File>
  --input               Input Contact Map    
  
```

## Tutorial: 
**Single-Channel**  
In this tutorial, you'll learn to test ContactGAN for single channel inputs.  
***For the purpose of this tutorial, please refer to example contact map input and output are provided in data/example_files/***   

### ContactGAN Usage  
To run ContactGAN, you will need an input contact map from one of the following 4 methods - CCMpred, DeepCov, DeepContact, or trRosetta.  
An example contact map can be found [here](https://github.com/kiharalab/ContactGAN/tree/master/data/example_files/input/single_channel).  
Model files required to run ContactGAN can be found [here](https://github.com/kiharalab/ContactGAN/tree/master/model/CCMPred)   
Once you have a contact map e.g. CCMpred, you can run ContactGAN as follows:  
1) Specify input map directory to --input argument
2) G_res_blocks - Number of Generator ResNet blocks. Specify 6 for trRosetta and 3 for others.  
3) D_res_blocks - Number of Disciminator ResNet blocks. Specify 3.  
4) G_path - Generator Model Path. If you're using CCMpred you can use this [path](https://github.com/kiharalab/ContactGAN/tree/master/model/CCMPred/Generator)  
5) D_path - Discriminator Model Path. If you're using CCMpred you can use this [path](https://github.com/kiharalab/ContactGAN/tree/master/model/CCMPred/Discriminator)  

```
python test/denoising_gan_test.py --input=data/example_files/input/sigle_channel --G_res_blocks=3 --D_res_blocks=3 --G_path=model/CCMPred/Generator --D_path=model/CCMPred/Discriminator

```

**Multi-Channel (2 channels)**  
In this tutorial, you'll learn to test ContactGAN for multi-channel (2) inputs.  
***For the purpose of this tutorial, please refer to example contact map input and output are provided in data/example_files/***   

### ContactGAN Usage  
To run ContactGAN with multiple channels i.e. either 2 or 3, you will need specify multiple contact maps as inputs.  
An example for multiple channels can be found [here](https://github.com/kiharalab/ContactGAN/tree/master/data/example_files/input/multi_channel).  
Model files required to run ContactGAN can be found [here](https://github.com/kiharalab/ContactGAN/tree/master/model/CCMPred_DeepContact)   
1) Specify input map directories i.e., channel1 and channel2 to --input argument.  
2) G_res_blocks - Number of Generator ResNet blocks. Specify 6 for trRosetta and 3 for others.  
3) D_res_blocks - Number of Disciminator ResNet blocks. Specify 3.  
4) G_path - Generator Model Path. If you're using CCMpred you can use this [path](https://github.com/kiharalab/ContactGAN/tree/master/model/CCMPred_DeepContact/Generator)  
5) D_path - Discriminator Model Path. If you're using CCMpred you can use this [path](https://github.com/kiharalab/ContactGAN/tree/master/model/CCMPred_DeepContact/Discriminator)  

```
python test/denoising_gan2_test.py --input data/example_files/input/multi_channel/channel1 data/example_files/input/multi_channel/channel2 --G_res_blocks=3 --D_res_blocks=3 --G_path=model/CCMPred_DeepContact/Generator --D_path=model/CCMPred_DeepContact/Discriminator

```
### Output contact map Visualization  
```
python util/plot_cmap.py --input=data/example_files/output/5OHQA.npy
 
```
Below is an example visualization for contact maps of CCMpred before and after ContactGAN for protein with PDB ID: [5OHQA](http://www.rcsb.org/structure/5OHQ).      
![](https://github.com/kiharalab/ContactGAN/blob/master/data/git/fig2.jpg)   
