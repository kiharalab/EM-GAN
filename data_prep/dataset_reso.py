#Run Inst.s
#python dataset.py trimmap emid output_dir


import sys
import os
import numpy as np
dataFile = sys.argv[1];
fil1 = open(dataFile,'r');


j=0


folderBase=sys.argv[3]
count=0
rows1=[]
rows2=[]
#dataDir=folderBase+dataFile+"_tmp"
dataDir=folderBase
if not os.path.exists(dataDir):
    os.system("mkdir "+dataDir)
for line in fil1:
    if(line.rstrip()==''):
        continue
    elif(line.startswith("#") or line.startswith("m")):
        continue   
    li = line.split(',');
    if(len(li)!=2*25*25*25):
        print("breaking")
        break

    rows1=np.array(li[0:25*25*25],dtype="float32")
    outputFile1 = dataDir+"/"+sys.argv[2]+"_LR"+str(count)
    np.save(outputFile1,rows1)
    rows2=np.array(li[25*25*25:2*25*25*25],dtype="float32")
    outputFile2 = dataDir+"/"+sys.argv[2]+"_HR"+str(count)
    np.save(outputFile2,rows2)    
    count+=1
    #print(count)
    #if count==32:
    #    break
fil1.close()