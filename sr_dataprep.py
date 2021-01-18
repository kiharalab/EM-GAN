import os
import sys
import subprocess
import tempfile
import numpy as np

trimmap_dir="./"
dataset_dir="./data_dir"
write_dir=os.path.join(dataset_dir,"sr_hr_lr")
flagPath = os.path.join(dataset_dir,"flag_dir")

if(not os.path.exists(write_dir)):
    os.mkdir(write_dir)

if(not os.path.exists(flagPath)):
    os.mkdir(flagPath)
def fil_process_new(filArr,trim,id):
	idx=0
	with open(trim,"r") as tfile:
		for line in tfile:

			if(line.rstrip()==''):
				continue
			elif(line.startswith("m")):
				continue   
			elif(line.startswith("#")):
				if line.startswith("#M1"):
					coords = line.strip().split()[2:5]
					# print(coords)

				else:
					continue
			elif(line.startswith("0.")):
				sr_orig = open(filArr[idx],"r")


				sr = open(os.path.join(write_dir,id+"_"+str(idx)+"_SR.situs"),"w")
				sr.write("1.000000 "+ coords[0]+" " +coords[1]+" "+coords[2]+" "+"25 25 25\n")
				sr.write(sr_orig.readline())
				sr.close()
				sr_orig.close()
				
				li = line.split(',')
				if(len(li)!=2*25*25*25):
					print("breaking")
					break				
				
				lr = open(os.path.join(write_dir,id+"_"+str(idx)+"_LR.situs"),"w")
				lr.write("1.000000 "+ coords[0]+" " +coords[1]+" "+coords[2]+" "+"25 25 25\n")
				lr.write(" ".join(li[0:25*25*25]))
				lr.close()

				hr = open(os.path.join(write_dir,id+"_"+str(idx)+"_HR.situs"),"w")
				hr.write("1.000000 "+ coords[0]+" " +coords[1]+" "+coords[2]+" "+"25 25 25\n")
				hr.write(" ".join(li[25*25*25:2*25*25*25]))
				hr.close()
				
				idx+=1
sr_imageFileNames = [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir) if "SR" in x]
print(dataset_dir)


map_mrc={}

for f in sr_imageFileNames:
	
	filname=f.split("/")[-1].split(".")[0]
	key = filname.split("_")[1]
	idx = int(filname.split("_")[2])
	# print(key)
	# print(idx)
	if key not in map_mrc:
		ARR = [0]*12100
		ARR[idx] = f
		map_mrc[key]=ARR
		
	else:
		map_mrc[key][idx] = f
for i in map_mrc:
	print(i)
	newflag=os.path.join(flagPath,i+".flag")
	print(newflag)
	if(not os.path.exists(newflag)):
		os.system("touch "+newflag)
		trimmap = os.path.join(trimmap_dir,i+"_trimmap")
		fil_process_new(map_mrc[i],trimmap,i)
		print(i+" done")