import os
import subprocess

map_mrc={}

def avg():
	for i in map_mrc:
		idx=map_mrc[i].index(0)
		w="\n".join(map_mrc[i][:idx])
		tmpF = open(os.path.join(flagPath,"tmpF"),"w")
		tmpF.write(w)
		tmpF.close()
		print('./MergeMap -l '+os.path.join(flagPath,"tmpF"))
		exe='./MergeMap -l '+os.path.join(flagPath,"tmpF")
		proc = subprocess.Popen(exe, shell=True)
		proc.wait()			
				
		# rc("open Merged.mrc")
		# rc("volume #0 save "+os.path.join(flagPath,i+"_chim.mrc"))
		# rc("close all")

		print(i+" done")
	# rc("stop now")

dataset_dir="./data_dir/sr_hr_lr/"
flagPath = "./"
sr_imageFileNames = [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir) if "SR" in x]

print(dataset_dir)
print(len(sr_imageFileNames))
for f in sr_imageFileNames:
	filname=f.split("/")[-1].split(".")[0]
	key = filname.split("_")[0]
	idx = int(filname.split("_")[1])
	if key not in map_mrc:
		ARR = [0]*12100
		ARR[idx] = f
		map_mrc[key]=ARR
		
	else:
		map_mrc[key][idx] = f
avg()

