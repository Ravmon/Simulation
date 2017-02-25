from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from math import *
from scipy.misc import imread
from skimage import data, img_as_float
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral
import pandas as pd



def make_sphere(xMat,yMat,x,y,r):
       return np.sqrt(((xMat-(x)))**2 +((yMat-(y)))**2) <= r

image_size_x = 400
image_size_y = 400    
rows=np.linspace(0,image_size_x,image_size_x) 
cols=np.linspace(0,image_size_y,image_size_y)
xMat,yMat=np.meshgrid(rows,cols)    

num_points_x,num_points_y=20,20 # Gives you number of spheres on each axis 10,10
x_points=range(1,num_points_x+1)
#print(list(x_points))
y_points=range(1,num_points_y+1)
#h,width=np.meshgrid(rows,cols)   
#n=10
#m=10
spheres=[]
xspace=image_size_x//(num_points_x+1) # gives equally spaced points(spheres) on the x axis on the grid(image)
yspace=image_size_y//(num_points_y+1)
for ix in x_points:
    for iy in y_points:
        coords=(ix*xspace,iy*yspace) #Dimensionally correct.points *xspace(units/points)
        spheres.append(coords)
spheres=np.array(spheres)     
#print(spheres)


distFromCenter = np.zeros((image_size_x, image_size_y))
radius=1.5

for c in spheres:
    
    distFromCenter += make_sphere(xMat, yMat, c[0], c[1], radius)


noisy_sphere = distFromCenter + 4.5 * distFromCenter.std() * np.random.random(distFromCenter.shape)   
plt.imshow(distFromCenter,cmap=plt.cm.gray)
#plt.imshow(distFromCenter,cmap=plt.cm.gray)
plt.imsave('no_noise.png',distFromCenter,cmap=plt.cm.gray) 
#plt.show()

np.random.seed(20)
noisy_sphere_g=distFromCenter +np.random.normal(2,.1,(400,400)) #0.25 
plt.imshow(noisy_sphere_g,cmap=plt.cm.gray)
#plt.imshow(noisy_sphere_g,cmap=plt.cm.gray)
plt.imsave('gaussian_noise.png',noisy_sphere_g,cmap=plt.cm.gray)
plt.show()
#plt.imshow(distFromCenter)
#plt.imshow(noisy_sphere)
#plt.imshow(noisy_sphere_g)
#plt.imsave('mess.png',noisy_sphere,cmap=plt.cm.jet)
#plt.imsave('mess.png',noisy_sphere)


#def Crystal_Stat(file): #Individual crystal statistics
#    data=pd.read_csv(file)
#    df=data.groupby('ImageNumber')# This takes each image number column and creates a data frame from it
#    group_MaxFD=df['AreaShape_MaxFeretDiameter']#This picks out my column of interest
#    MaxFD=[] #MaxFD=Maximum Fereit Diameter
#    for name,group in group_MaxFD:
#        MaxFD.append(group)
#        Ferit_Diameter=np.array(MaxFD)
#    return Ferit_Diameter  
#    
#Sample5=Crystal_Stat('/home/raphael/Documents/CellProfiler/ValidatingCP/SMALLNuclei.csv' )
##binwidth=0.26#crystal.csv
#binwidth=.65
##bins=(np.max(Sample5)-np.min(Sample5))/binwidth
#bins=np.arange(np.min(Sample5), np.max(Sample5) + binwidth, binwidth)
##plt.hist(Sample5.T, bins=6, range=None, normed=False)
##bins,np.max(Sample5)-np.min(Sample5)
#plt.hist(Sample5.T,bins=bins,normed=False)
#plt.show()   
