# SuperEM
SuperEM is a 3D deep learning based super-resolution method which uses Generative Adversarial Networks (GAN) to improve the resolution of cryo-EM maps in the resolution ranges of 3 Å to 6 Å.  
Copyright (C) 2020 Sai Raghavendra Maddhuri Venkata Subramaniya, Genki Terashi, Daisuke Kihara, and Purdue University.

License: GPL v3 for academic use. (For commercial use, please contact us for different licensing)

Contact: Daisuke Kihara (dkihara@purdue.edu)

Cite : Sai Raghavendra Maddhuri Venkata Subramaniya, Genki Terashi & Daisuke Kihara. Super Resolution Cryo-EM Maps With 3D Deep Generative Networks. In submission (2020).  

## About SuperEM  
>   
![](https://github.com/kiharalab/SuperEM/blob/master/data/git/architecture.png)   


## Pre-required software
```

Python 3 : https://www.python.org/downloads/  
pytorch : pip/conda install pytorch   

```
## Instructions  
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="data/git/w3.css">  
<link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Lato">
<link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Montserrat">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">

<div id="commands" class="w3-row-padding w3-padding-64 w3-container">
  <div class="w3-content">
	<h5> In the following section, we give a step-by-step guide to run various programs that we talked about in the earlier section.
	You can find the link for these programs in Downloads tab.
  </h5>

	<div id="datagen" class="w3-row-padding w3-padding-64 w3-container">
	<h2>Input file generation</h2>
	<h5>
	Generate the input file called [your_map_id]_dataset from your map file by following the below 2 steps.
	</h5>
	<b><h3>1) Trimmap File generation</h3></b>
      <pre><p class="w3-code">
<b>data_prep/HLmapData_new [sample_mrc] [options] > [output_trimmap_filename]</b><br>
<b>INPUTS:</b><br>
HLmapData_new expects sample_mrc to be a valid filename. Supported file formats are Situs, CCP4, and MRC2000. 
Format is deduced from FILE's extension. 
<br>
<b>OPTIONS:</b><br>
-c, --contour The level of isosurface to generate density values for. 
You can use the author recommended contour level for experimental EM maps.
<i>default=0.0</i><br>
-w [integer] This option sets the dimensions of sliding cube used for input data generation.
The size of the cube is calculated as <i>2*w+1</i>.
We recommend using a value of 12 for this option that generates input cube of size 25*25*25.
Please be mindful while increasing this option as it increases the portion of an EM map a single cube covers.
Increasing this value also increases running time.  
<i>default=12 (->25x25x25)</i><br>
-h, --help, -?, /? Displays the list of above options.<br><br>
<b>USAGE:</b>
./HLmapData_new protein.map -c 0.0 -w 12 >  protein_trimmap
      </p></pre>
<br>

<b><h3>2) Input dataset file generation</h3></b>
	<h5>
	This program is used to generate input dataset file from the trimmap file generated in step 1.<br>
	This program is a python script and it works with both python2 and python3. They can be downloaded <a href=https://www.python.org/downloads/ target="_blank">here</a>.<br>
	</h5>
      <pre><p class="w3-code">
<b>python data_prep/dataset_final.py [sample_trimmap] [input_dataset_file] [ID]</b>
<br>
<b>INPUTS:</b> 
Inputs to this script are trimmap, and ID is a unique identifier of a map such as 
SCOPe ID, EMID, etc.<br><br>
<b>OUTPUT:</b>
Specify a name for input dataset file in place of [input_dataset_file].<br><br>
<b>USAGE:</b>
python data_prep/dataset_final.py protein_trimmap protein_dataset protein_id
     </p></pre>  
      </div>
  </div>
    <div class="w3-content">
	<div id="superem" class="w3-row-padding w3-padding-64 w3-container">
	<h2>Super Resolution EM map generation</h2>
	<h5>
	Run SuperEM program for generating super-resolution em maps from low-resolution experimental maps.<br>
	Use <b>test.py</b> to generate super-resolution map.<br>
	This program is a python script and it works with both python2 and python3.
	</h5>
      <pre><p class="w3-code">
<b>python test.py --input=INPUT_EM_MAP_DIR --G_res_blocks=15 --D_res_blocks=3 --G_path=GENERATOR_MODEL_PATH --D_path=DISCRIMINATOR_MODEL_PATH</b><br>
<b>INPUT:</b>
  --input               Path to directory containing input EM map files  
  --G_res_blocks        Number of ResNet blocks in Generator (Default : 15)
  --D_res_blocks        Number of ResNet blocks in Discriminator (Default : 3)
  --G_path              Specify path of Generator model
  --D_path              Specify path of Discriminator model
<br>
<b>OUTPUT:</b>
This program writes output super-resolution em map to the same directory as input.

<b>USAGE:</b>
python test.py --input=INPUT_EM_MAP_DIR --G_res_blocks=15 --D_res_blocks=3 --G_path=model/Generator --D_path=model/Discriminator
      </p></pre>
      </div>
  </div>

</div>
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
