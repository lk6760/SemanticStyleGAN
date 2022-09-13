import os
import shutil

cwd = os.path.join(os.getcwd(),"results")

im_dir = "results_im"

if os.path.exists(im_dir):
    shutil.rmtree(im_dir)
os.makedirs(os.path.join(im_dir), exist_ok=True)

#shutil.copytree(src, dest) 

for i in sorted(os.listdir(cwd)):
    path = os.path.join(cwd,i)
    for j in os.listdir(path):
        if "recon" in j:
            src = os.path.join(path,j)
            dest = src[:48]+"_im"+src[48:]
            print(src,"\n",dest)
            shutil.copytree(src, dest)
    


