import sys
import os
import numpy as np
import subprocess
import tempfile

from scipy import ndimage
import numpy as np
import mrcfile

infFile = "/net/kihara/home/smaddhur/tensorFlow/SuperReso/map_pdb_information"
vmd_file="/net/kihara/home/smaddhur/tensorFlow/vmd/cross_corr_SuperRes_2_6_final_65"
sitPath = "/net/kihara/home/han273/em_project/dataset/map/"
newPath="/net/kihara/home/smaddhur/tensorFlow/SuperReso/EM_data/SIM_MAP"
newSitusPath="/net/kihara/home/smaddhur/tensorFlow/SuperReso/EM_data/EXP_MAP"
pdbPath="/net/kihara/home/smaddhur/tensorFlow/PDB_SAI/"

trimPath = "/net/kihara-fast-scratch/smaddhur/SuperReso/EM_data/trimmap_new_2_6/"
flagPath= "/net/kihara-fast-scratch/smaddhur/SuperReso/EM_data/flag_dir/"
# dataPath = "/net/kihara/scratch/smaddhur/SuperReso/dataset_large/"
dataPath = "/net/kihara-fast-scratch/smaddhur/SuperReso/EM_data/dataset_new_2_6/"


if(not os.path.exists(trimPath)):
    os.system("mkdir "+trimPath)

if(not os.path.exists(dataPath)):
    os.system("mkdir "+dataPath)
    
fil1 = open(infFile,'r')
pdbArr = []
mapArr = []
shArr=[]
conArr=[]
resArr=[]
lines1 = fil1.readlines()
for line in lines1:
    w = line.split()
    if(len(w)>=5 and w[3]!='None' and float(w[3])>=2.0 and float(w[3])<=6.0 and float(w[2])>=1.0):
        pdbArr.append(w[1])
        mapArr.append(w[0])
        resArr.append(float(w[3]))
        shArr.append(w[2])
        conArr.append(w[4])



# print(pdbArr)
# print(len(mapArr))
fil1.close()
fil1 = open(vmd_file,'r')
lines1 = fil1.readlines()


pdbArrU = []
mapArrU = []
shArrU=[]
conArrU=[]
resArrU=[]
for line in lines1:
    w = line.split()
    [ma,pd]=w[0].split("_")
    if ma not in mapArr:
        # print(ma)
        continue
    # if resArr[mapArr.index(ma)] <= 3.5:
    #     print(ma)
    mapArrU.append(ma)
    pdbArrU.append(pd)
    shArrU.append(shArr[mapArr.index(ma)])
    conArrU.append(conArr[mapArr.index(ma)])
    resArrU.append(resArr[mapArr.index(ma)])

# print(len(mapArrU))
# print(pdbArrU)
fil1.close()
x=0
i=0
count=0
                        
pdbs_final=['3g37', '3izk', '3j03', '3j3x', '3j5m', '3j5p', '3j63', '3j6e', '3j6f', '3j6g', '3j6j', '3j89', '3j8a', '3j8g', '3j8i', '3j94', '3j9d', '3j9i', '3j9l', '3j9p', '3ja7', '3jad', '3jar', '3jb7', '3jbt', '3jbv', '3jbw', '3jc5', '3jca', '3jcf', '3jck', '3jcl', '3jcn', '3jcu', '4cc8', '4v19', '4v1w', '5a0q', '5a22', '5a2t', '5a5t', '5adx', '5aey', '5aj3', '5an8', '5fj9', '5fja', '5flu', '5fn3', '5fwk', '5fyw', '5g05', '5g06', '5g2x', '5gae', '5gah', '5gao', '5gaq', '5go9', '5goa', '5grs', '5gw5', '5h1b', '5h3o', '5h64', '5hi9', '5iv5', '5jb3', '5jco', '5ju8', '5jul', '5jzw', '5kcr', '5ken', '5key', '5kne', '5kuf', '5lcw', '5ld2', '5ldw', '5lij', '5ljo', '5ljv', '5lks', '5ll6', '5lmo', '5lmp', '5lmq', '5lmr', '5lms', '5lmt', '5lmu', '5lmv', '5lmx', '5lqp', '5lzc', '5lzd', '5lze', '5lzf', '5lzp', '5m0q', '5m50', '5m5x', '5m5y', '5mdv', '5mkf', '5mmi', '5mq3', '5mqf', '5mz6', '5n5n', '5n8o', '5n8y', '5n9y', '5nj3', '5no3', '5no4', '5np1', '5nsi', '5nss', '5o2r', '5o66', '5oa1', '5oaf', '5oej', '5of4', '5ofo', '5ogc', '5szs', '5t15', '5t4d', '5tcp', '5tcq', '5tj5', '5tqw', '5tr1', '5ttp', '5u0a', '5u0p', '5u1c', '5u6o', '5udb', '5uk0', '5uk1', '5uk2', '5uot', '5up2', '5up6', '5upa', '5upc', '5upw', '5uvn', '5uz4', '5uz9', '5v7q', '5v8l', '5v8m', '5vc7', '5vkq', '5vn8', '5vrf', '5vxx', '5vy8', '5vy9', '5vya', '5w0s', '5w1r', '5w5y', '5w68', '5wc0', '5wc3', '5wda', '5weo', '5wfe', '5wj5', '5wq7', '5wq9', '5wrg', '5x0y', '5x5b', '5x5c', '5x5f', '5x8r', '5x8t', '5xb1', '5xlr', '5xmi', '5xmk', '5xml', '5xte']
pdbArrFin = []
mapArrFin = []
shArrFin=[]
conArrFin=[]
resArrFin=[]

for i in pdbs_final:
    idx=pdbArrU.index(i)

    mapArrFin.append(mapArrU[idx])
    pdbArrFin.append(pdbArrU[idx])
    shArrFin.append(shArrU[idx])
    conArrFin.append(conArrU[idx])
    resArrFin.append(resArrU[idx])



# print((mapArrFin[176]))
# print((pdbArrFin[176]))
# print((shArrFin[176]))
# print((conArrFin[176]))
# print((resArrFin[176]))

mapArrU=mapArrFin
pdbArrU=pdbArrFin
shArrU=shArrFin
conArrU=conArrFin
resArrU=resArrFin
# print(len(mapArrU))

# c=0
# mapArrExp=[]
# for fil in mapArrU:
#     if fil.startswith("1"):
#         continue
#     newsitus=os.path.join(newSitusPath,fil+".mrc")
#     i=mapArrU.index(fil)
#     sh=float(shArrU[i])
#     print(sh)
#     situs=os.path.join(sitPath,"emd_"+fil+".map")
#     if os.path.exists(situs):
#         with mrcfile.open(situs) as mrc:
#             nx,ny,nz,nxs,nys,nzs,mx,my,mz = mrc.header.nx,mrc.header.ny,mrc.header.nz,mrc.header.nxstart,mrc.header.nystart,mrc.header.nzstart,mrc.header.mx,mrc.header.my,mrc.header.mz
#             # nxs,nys,nzs = mrc.header.nxstart,mrc.header.nystart,mrc.header.nzstart
#             if nxs != 0 or nys !=0 or nzs !=0:
#                 c+=1
#                 continue

#             mapArrExp.append(fil)
#             orig=mrc.header.origin
#             data = np.array(mrc.data,dtype="float32")
#             # data=np.swapaxes(data,0,2)
#             # data_new=data
#             data_new=np.zeros((nx,ny,nz))
#             data_new = ndimage.interpolation.zoom(data, (sh,sh,sh),order=3)
#             # data_new=np.swapaxes(data_new,0,2)
#             mrc_new = mrcfile.new(newsitus,data=data_new, overwrite=True)
#             vsize=mrc_new.voxel_size
#             vsize.flags.writeable = True
#             vsize.x=1.0
#             vsize.y=1.0
#             vsize.z=1.0
#             mrc_new.voxel_size=vsize
#             mrc_new.update_header_from_data()   
#             mrc_new.header.nxstart=nxs*sh
#             mrc_new.header.nystart=nys*sh
#             mrc_new.header.nzstart=nzs*sh
#             mrc_new.header.origin=orig
#             mrc_new.update_header_stats()
#             #print(mrc_new.data.shape)
#             #print(mrc_new.voxel_size)
#             mrc.print_header()
#             mrc_new.print_header()
#             mrc_new.close()
#             del data
#             del data_new
#     # break
# print(mapArrExp)
# print(len(mapArrExp))


mapArrExp = ['5779', '5925', '6123', '5623', '2871', '6344', '6351', '6481', '6486', '6489', '6535', '6441', '6551', '6479', '6526', '3285', '2484', '2788', '6337', '3056', '3179', '3180', '3238', '3337', '3378', '3388', '3366', '3331', '8001', '8004', '8013', '8015', '9528', '9529', '9537', '9566', '6656', '6668', '6618', '3374', '8148', '8150', '8176', '8177', '8188', '8237', '8242', '8267', '8289', '4037', '4038', '4040', '4054', '4061', '4062', '4070', '4071', '4074', '4075', '4076', '4077', '4078', '4079', '4080', '4083', '4088', '4098', '4123', '4124', '4125', '4126', '4128', '4138', '4154', '3447', '3448', '3489', '3524', '3531', '3543', '3545', '3583', '3589', '3601', '3602', '3605', '3654', '3662', '3663', '3672', '3696', '3730', '8640', '3727', '3773', '3790', '3802', '3776', '8331', '8342', '8354', '8398', '8399', '8409', '8436', '8454', '8467', '8479', '8481', '8511', '8540', '8581', '8584', '8585', '8586', '8595', '8608', '8621', '8624', '8641', '8643', '8644', '8658', '8702', '8717', '8728', '8744', '8745', '8746', '8750', '8751', '8771', '8778', '8794', '8795', '8823', '8840', '6675', '6677', '6679', '6700', '6710', '6711', '6714', '6732', '6733', '6734', '6735', '6774']
pdbArrExp = []
shArrExp = []
conArrExp = []
resArrExp = []

for i in mapArrExp:
    idx=mapArrU.index(i)

    # mapArrFin.append(mapArrU[idx])
    pdbArrExp.append(pdbArrU[idx])
    shArrExp.append(shArrU[idx])
    conArrExp.append(conArrU[idx])
    resArrExp.append(resArrU[idx])

mapArrU=mapArrExp
pdbArrU=pdbArrExp
shArrU=shArrExp
conArrU=conArrExp
resArrU=resArrExp

# print(len(pdbArrU))
# print((mapArrU[76]))
# print((pdbArrU[76]))
# print((shArrU[76]))
# print((conArrU[76]))
# print((resArrU[76]))

# print(pdbArrU)
# print(len(pdbArrU))


########Print Final PDBs
tmpPdb=[]
tmpMap=[]
exclusionList=["8812","8767","8739","8287","8150","6617","6240","6239","6057","5925","4154","3803","3778","3589","3444","3020"]

for fil in mapArrU:
    i=mapArrU.index(fil)
    pdb = pdbArrU[i]+".pdb"
    pdbPath1=os.path.join(pdbPath,pdb)
    if os.path.exists(pdbPath1) and fil not in exclusionList:
        tmpPdb.append(pdbArrU[i])
        tmpMap.append(mapArrU[i])

print(tmpMap)
print(tmpPdb)
print(len(tmpPdb))
exit()
#####################

# for fil in mapArrU:

#     i=mapArrU.index(fil)
#     pdb = pdbArrU[i]+".pdb"
#     pdbPath1=os.path.join(pdbPath,pdb)

#     newflag=os.path.join(newPath,fil+".flag")
#     if(not os.path.exists(newflag)):
#         fil_f = open(newflag,'w')
#         fil_f.close()

#         newpdbPath=pdbPath1
#         resVal="3.0"
#         if(resArrU[i] <= 3.5):
#             resVal="1.8"
            
        
#         # print(newpdbPath)
#         print("1\n1\n1\n-"+str(resVal)+"\n1\n1\n1\n")
#         wFile=open("tmp_in_"+fil,"w")

#         wFile.write("1\n1\n1\n-"+str(resVal)+"\n1\n1\n1\n")
#         wFile.close()
        
# #         print('echo '+lin+" | "+'pdb2vol ' + newpdbPath + " " + newPath+fil+".mrc")
#         exe='cat tmp_in_'+fil+" | "+'pdb2vol ' + newpdbPath + " " + os.path.join(newPath,fil+".mrc")
#         print(exe)
#         proc = subprocess.Popen(exe, shell=True)
#         proc.wait()      
#         proc1= subprocess.Popen('rm tmp_in_'+fil, shell=True)
#         proc1.wait()   
#     # break     







# j=0        
# for fil in mapArrU:
#     filid = fil
#     i=mapArrU.index(fil)
#     # if(fil!="5779"):
#     #    continue
#     # print(fil)
#     situs1=os.path.join(newSitusPath,fil+".mrc")
#     situs2=os.path.join(newPath,fil+".mrc")
#     exclusionList=["8812","8767","8739","8287","8150","6617","6240","6239","6057","5925","4154","3803","3778","3589","3444","3020"]
#     # if fil in exclusionList:
#     #     print(fil)
#     if(os.path.exists(situs1) and os.path.exists(situs2) and fil not in exclusionList):
#         j+=1
#         newflag=os.path.join(flagPath,fil+".flag")
#         print(newflag)
#         if(not os.path.exists(newflag)):
#             os.system("touch "+newflag)
#             print(flagPath + fil+"_trimmap")
#             #print('./HLmapData -a ' + situs1 + ' -b ' + situs2 + ' -A ' + conArrU[i]+ ' -B 0 -w 12 -s 4 > ' + trimPath + fil+"_trimmap")
#             x = os.system('./HLmapData -a ' + situs1 + ' -b ' + situs2 + ' -A ' + conArrU[i]+ ' -B 0 -w 12 -s 5 > ' + os.path.join(trimPath,fil+"_trimmap"));
#             #python dataset_reso.py 8242_trimmap 8242data ./
#             print('python dataset_reso.py '+trimPath + fil+"_trimmap "+ fil+"_data "+dataPath)
#             os.system('python dataset_reso.py '+trimPath + fil+"_trimmap "+ fil+"_data "+dataPath)
#             print("done for situs : %s",(situs1))
#             print("done for j:"+str(j))
#             # break
# print(j)
