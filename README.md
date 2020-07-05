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
	<h5> In the following section, we give a step-by-step guide to run various programs for SuperEM.
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

## Tutorial: 
<div id="ex2" class="w3-row-padding w3-padding-32 w3-container">
  <div class="w3-content">
   <h1>Experimental map example (EMD-2788)</h1>
    <div class="w3-twothird">

	
<h5>You can download the EM map for protein structure with EMID 2788 <a href="ftp://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-2788" target="_blank" >here</a>. Use this map file and follow the instructions in step 1 of usage guide to generate input dataset file.
		The trimmap file is generated as 
	</h5>
	<pre><p class="w3-code">data_prep/HLmapData_new  2788.mrc -c 0.16 >  2788_trimmap</p></pre>
	<h5>The author recommended contour level for the map EMD-2788 is 0.16 which has been provided as one of the options above.
	</h5>
	<h5>You can generate the input dataset file as follows,
	</h5>
	<pre><p class="w3-code">python data_prep/dataset_final.py 2788_trimmap 2788_dataset 2788</p></pre>
	<h5>	
	If the generated input file is 2788_dataset, write the file location to a dataset location file as follows
      </h5>
	  <pre><p class="w3-code">echo ./2788_dataset > test_dataset_location</p></pre>
    </div>
    <div class="w3-third w3-center ">
     <img src=data/git/1.png width="600" height="300"> <p align="left">Density Map, 2788.mrc</p>
    </div>
	<!--style="width:150%;height:500%"-->

  </div>
</div>
<div class="w3-content">
    <div class="w3-twothird">
     <h3>SuperEM super-resolution map generation </h3>
    <h5>
	You can then run the SuperEM program to generate super-resolution EM map as follows
      </h5>
<p class="w3-code"> python test.py --input=test_dataset_location --G_res_blocks=15 --D_res_blocks=3 --G_path=model/Generator --D_path=model/Generator </p>
		<h5>
	An example of generated super-resolution map of EMD-2788 is shown on the right.
  </div> 
    <div class="w3-third w3-center ">
     <img src=data/git/2.png width="600" height="300">  <p align="left"> Super-Resolution (SR) Map, 2788_SR.mrc</p>
    </div>  
</div>
<div class="w3-row-padding w3-padding-32 w3-container">
</div>
