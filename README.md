# EM-GAN


<a href="https://github.com/marktext/marktext/releases/latest">
   <img src="https://img.shields.io/badge/EMGAN-v1.0.0-green">
   <img src="https://img.shields.io/badge/platform-Linux%20%7C%20Windows%20%7C%20Mac%20-green">
   <img src="https://img.shields.io/badge/Language-python3-green">
   <img src="https://img.shields.io/badge/Language-C-green">
   <img src="https://img.shields.io/badge/dependencies-tested-green">
   <img src="https://img.shields.io/badge/licence-GNU-green">
</a>      <br>
 
EM-GAN is a computational tool, which enables capturing protein structure information from cryo-EM maps more effectively than raw maps. It is based on 3D deep learning. It is aimed to help protein structure modeling from cryo-EM maps.  <br>
Copyright (C) 2021 Sai Raghavendra Maddhuri Venkata Subramaniya, Genki Terashi, Daisuke Kihara, and Purdue University.  

License: GPL v3 for academic use. (For commercial use, please contact us for different licensing)  

Contact: Daisuke Kihara (dkihara@purdue.edu)  

Cite : Sai Raghavendra Maddhuri Venkata Subramaniya, Genki Terashi & Daisuke Kihara. Improved Protein Structure Modeling Using Enhanced Cryo-EM Maps With 3D Deep Generative Networks. Bioinformatics, in press (2023).  

Google Colab: https://tinyurl.com/3ccxpttx

## About EM-GAN  

An increasing number of biological macromolecules have been solved with cryo-electron microscopy (cryo-EM). Over the past few years, the resolutions of density maps determined by cryo-EM have largely improved in general. However, there are still many cases where the resolution is not high enough to model molecular structures with standard computational tools. If the resolution obtained is near the empirical border line (3 - 4 Å), improvement in the map quality facilitates improved structure modeling. Here, we report that protein structure modeling can often be substantially improved by using a novel deep learning-based method that prepares an input cryo-EM map for modeling. The method uses a three-dimensional generative adversarial network, which learns density patterns of high and low-resolution density maps.   

GAN architecture of EM-GAN is shown below.  

>   
![](https://github.com/kiharalab/EM-GAN/blob/master/data/git/architecture.png)   


## Pre-required software
```

Python 3 : https://www.python.org/downloads/  
pytorch : pip/conda install pytorch   

```
## Dependencies
```
mrcfile==1.2.0
numpy>=1.19.4
numba>=0.52.0
torch>=1.6.0
scipy>=1.6.0
```
## Availability
This software is free to use under GPL v3 for academic use. For commercial use, please contact us for different licensing.   

## Timings
Please allow 30 mins on average to get the output, since 3D input processing and inferencing takes some time.
Our running time is directly correlated to the size of the structures. For example, a map with 260 * 260 * 260 can take 2 hours to finish.  

## System requirements and compatibility
OS: Any (e.g CentOS, Linux, Windows, Mac).  
Necessary libararies: Please refer to the dependencies above and make sure that they're installed.  
GPU: Optional (Any GPU with >4GB RAM should enable faster computation).

## Preparing the input map
If the cryo-EM map grid spacing is not 1, please modify the grid spacing to 1 by [ChimeraX](https://www.rbvi.ucsf.edu/chimerax/) as follows:
```
1 open your map via chimeraX.
2 In the bottom command line to type command: vol resample #1 spacing 1.0
3 In the bottom command line to type command: save newmap.mrc model #2
4 Then you can use the resampled map as an input map.
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
	<h5> In the following section, we give a step-by-step guide to run various programs for EM-GAN.
  </h5>

<div id="datagen" class="w3-row-padding w3-padding-64 w3-container">
<h2>Input file generation</h2>
<h5>
Generate the input file called [your_map_id]_dataset from your map file by following the below 2 steps.
</h5>
<b><h3>1) Trimmap File generation</h3></b>
<pre><p class="w3-code">
<b>data_prep/HLmapData -a [sample_mrc] -b  [sample_mrc] [options] > [output_trimmap_filename]</b><br>
<b>INPUTS:</b><br>
HLmapData_new expects sample_mrc to be a valid filename. Supported file formats are Situs, CCP4, and MRC2000. 
Format is deduced from FILE's extension. 
<br>
<b>OPTIONS:</b><br>
-a [mrc] Input map file of the experimental map.
<br>
-b [mrc] Input map file of the experimental map (Same map as above). If you have a simulated map available and are validating, specify that instead 
<br>
-A [float] The level of isosurface to generate density values for the first map (map specified with option -a). 
You can use the author recommended contour level for experimental EM maps.
<i>default=0.0</i><br>
-B  [float] The level of isosurface to generate density values for the first map (map specified with option -b)
You can use the author recommended contour level for experimental EM maps. If input is simulated map, specify 0.0
<i>default=0.0</i><br>
-w [integer] This option sets the dimensions of sliding cube used for input data generation.
The size of the cube is calculated as <i>2*w+1</i>.
We recommend using a value of 12 for this option that generates input cube of size 25*25*25.
Please be mindful while increasing this option as it increases the portion of an EM map a single cube covers.
Increasing this value also increases running time.  
<i>default=5 (->11x11x11)</i><br>
-s [integer] This option specifies the stride value to be used while generating input cubes
We recommend using a value of 4 for this option. Increasing this value also increases running time.  
<i>default=1 </i><br>
-h, --help, -?, /? Displays the list of above options.<br><br>
<b>USAGE:</b>
data_prep/HLmapData -a protein.mrc -b protein.mrc -A <Recommended contour level> -B <Recommended contour level> -w 12 -s 4 >  protein_trimmap
      </p></pre>
<br>

<b><h3>2) Input dataset file generation</h3></b>
	<h5>
	This program is used to generate input dataset file from the trimmap file generated in step 1.<br>
	This program is a python script and it works with both python2 and python3. They can be downloaded <a href=https://www.python.org/downloads/ target="_blank">here</a>.<br>
	</h5>
      <pre><p class="w3-code">
<b>python data_prep/generate_input.py [sample_trimmap] [<ID>_data] [dataset_folder]</b>
<br>
<b>INPUTS:</b> 
Inputs to this script are trimmap generated in the previous step, ID is a unique identifier of a map such as EMID, and dataset_folder which is a folder to write dataset files. <br><br>
<b>USAGE:</b>
python data_prep/generate_input.py protein_trimmap 1_data ./data_dir/
     </p></pre>  
      </div>
  </div>
    <div class="w3-content">
	<div id="superem" class="w3-row-padding w3-padding-64 w3-container">
	<h2>GAN-Modified EM map generation</h2>
	<h5>
	Run EM-GAN program for generating GAN-modified em maps from low-resolution experimental maps.<br>
	Use <b>test.py</b> to generate modified map.<br>
	This program is a python script and it works with both python2 and python3.
	</h5>
      <pre><p class="w3-code">
<b>python test.py --dir_path=INPUT_DATA_DIR --res_blocks=5 --batch_size=128 --in_channels=32 --G_path=GENERATOR_MODEL_PATH --D_path=DISCRIMINATOR_MODEL_PATH</b><br>
<b>INPUT:</b>
  --dir_path            Path to data directory created in the last step  
  --G_path              Specify path of Generator model
  --D_path              Specify path of Discriminator model
<br>
<b>OUTPUT:</b>
This program writes output modified em map cubes to the same directory as input.

<b>USAGE:</b>
python test.py --res_blocks=5 --batch_size=128 --in_channels=32 --G_path=model/G_model --D_path=model/D_model --dir_path=data_dir/
      </p></pre>

<h5>
Finally, run the below two python scripts to merge the generated cubes generated into a final modified map
</h5>
python sr_dataprep.py
      </p></pre>
python avg_model.py
      </p></pre>      
      </div>
  </div>
<h5>
Modified map is written to Merged.mrc file
</h5>

</div>

## Tutorial: 
<div id="ex2" class="w3-row-padding w3-padding-32 w3-container">
  <div class="w3-content">
   <h1>Experimental map example (EMD-2788)</h1>
    <div class="w3-twothird">

	
<h5>The example map of 2788 is available here from data/2788.mrc Use this map file and follow the instructions in step 1 of usage guide to generate input dataset file.
		The trimmap file is generated as 
	</h5>
	<pre><p class="w3-code">data_prep/HLmapData -a 2788.mrc -b 2788.mrc -A  0.16 -B 0.16 -w 12 -s 4 >  2788_trimmap</p></pre>
	<h5>The author recommended contour level for the map EMD-2788 is 0.16 which has been provided as one of the options above.
	</h5>
	<h5>You can generate the input dataset file as follows,
	</h5>
	<pre><p class="w3-code">python data_prep/generate_input.py 2788_trimmap 2788_data ./data_dir</p></pre>
    </div>
    <div class="w3-third w3-center ">
     <img src=data/git/1.png width="600" height="300"> <p align="left">Density Map, 2788.mrc</p>
    </div>
  </div>
</div>
<div class="w3-content">
    <div class="w3-twothird">
     <h3>EM-GAN modified map generation </h3>
    <h5>
	You can then run the EM-GAN program to generate modified EM map as follows
      </h5>

```
python test.py --res_blocks=5 --batch_size=128 --in_channels=32 --G_path=model/G_model --D_path=model/D_model --dir_path=./data_dir/
```
```
python sr_dataprep.py
```
```
python avg_model.py
```
<h5>
Modified map is written to Merged.mrc file
</h5>
<h5>
	An example of generated modified map of EMD-2788 is shown below.
  </div> 
    <div class="w3-center ">
     <img src=data/git/2.png width="600" height="300">  <p align="left"> GAN-modified Map, 2788_SR.mrc</p>
    </div>  
</div>
<div class="w3-row-padding w3-padding-32 w3-container">
</div>
