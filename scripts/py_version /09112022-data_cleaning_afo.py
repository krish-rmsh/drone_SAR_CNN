#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# In this notebook I plan on exploring the dataset afo


# In[12]:


import os
import glob


# In[13]:


# Mac local path
root_path =  r'/Users/krish/ljmu/1.data/afo/'
# result_root = root_path + '3.results/09112022-testing-code/'
# Path(result_root).mkdir(parents=True, exist_ok=True)


# In[14]:


# Lets study the class distribution i.e number of figures with and without humans
images_with_humans = 0
images_no_humans = 0
for annotation in glob.glob(root_path+'1category_labels/*.txt'):
    with open(annotation,'r') as fp:
        number_of_human = len(fp.readlines())
    if number_of_human == 0:
        images_no_humans+=1
    else:
        images_with_humans+=1
    


# In[15]:


print(images_no_humans,images_with_humans)
# Severe class imbalance, but dont forget tiling changes everything


# #### Lets tile and then study the same 

# In[16]:


import pandas as pd
import numpy as np
from PIL import Image
from shapely.geometry import Polygon, Point
from matplotlib import pyplot as plt
import glob
from pathlib import Path


# In[20]:


### Not sure what size should I tile to ..... The images are of better quality, so the objects are much smaller in size
# get all image names
imnames = glob.glob('/Users/krish/ljmu/1.data/afo/images/*.jpg')
#imnames = glob.glob('/content/drive/MyDrive/ljmu/1.data/swimmers_dataset/krish_train/*.jpg')
# specify path for a new tiled dataset
newpath = '/Users/krish/ljmu/1.data/afo/tiled/ts/'
#newpath = '/content/drive/MyDrive/ljmu/1.data/sampletiled/ts'
falsepath = '/Users/krish/ljmu/1.data/afo/tiled/false/'
#falsepath = '/content/drive/MyDrive/ljmu/1.data/sampletiled/false'
# python program to check if a path exists
#if path doesn’t exist we create a new path

#creating a new directory called pythondirectory
Path(newpath).mkdir(parents=True, exist_ok=True)
Path(falsepath).mkdir(parents=True, exist_ok=True)

# specify slice width=height
slice_size = 500

# tile all images in a loop

for imname in imnames:
    im = Image.open(imname)
    imr = np.array(im, dtype=np.uint8)
    height = imr.shape[0]
    width = imr.shape[1]
    labname = imname.split('/')[-1].replace('.jpg', '.txt')
    labels = pd.read_csv(root_path+'1category_labels/'+labname, sep=' ', names=['class', 'x1', 'y1', 'w', 'h'])
    
    # we need to rescale coordinates from 0-1 to real image height and width
    labels[['x1', 'w']] = labels[['x1', 'w']] * width
    labels[['y1', 'h']] = labels[['y1', 'h']] * height
    
    boxes = []
    #print(labels)
    # convert bounding boxes to shapely polygons. We need to invert Y and find polygon vertices from center points
    for row in labels.iterrows():
        x1 = row[1]['x1'] - row[1]['w']/2
        y1 = (height - row[1]['y1']) - row[1]['h']/2
        x2 = row[1]['x1'] + row[1]['w']/2
        y2 = (height - row[1]['y1']) + row[1]['h']/2

        boxes.append((int(row[1]['class']), Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])))
    
    counter = 0
    print('Image:', imname)
    # create tiles and find intersection with bounding boxes for each tile
    for i in range((height // slice_size)):
        for j in range((width // slice_size)):
            x1 = j*slice_size
            y1 = height - (i*slice_size)
            x2 = ((j+1)*slice_size) - 1
            y2 = (height - (i+1)*slice_size) + 1

            pol = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
            imsaved = False
            slice_labels = []

            for box in boxes:
                if pol.intersects(box[1]):
                    inter = pol.intersection(box[1])        
                    
                    if not imsaved:
                        sliced = imr[i*slice_size:(i+1)*slice_size, j*slice_size:(j+1)*slice_size]
                        sliced_im = Image.fromarray(sliced)
                        filename = imname.split('/')[-1]
                        ##filename = imname
                        slice_path = newpath + filename.replace('.jpg', f'_{i}_{j}.jpg')
                        
                        slice_labels_path = newpath + filename.replace('.jpg', f'_{i}_{j}.txt')
                        
                        print(slice_path)
                        #print('hi')
                        sliced_im.save(slice_path)
                        imsaved = True                    
                    
                    # get the smallest polygon (with sides parallel to the coordinate axes) that contains the intersection
                    new_box = inter.envelope 
                    
                    # get central point for the new bounding box 
                    centre = new_box.centroid
                    
                    # get coordinates of polygon vertices
                    x, y = new_box.exterior.coords.xy
                    
                    # get bounding box width and height normalized to slice size
                    new_width = (max(x) - min(x)) / slice_size
                    new_height = (max(y) - min(y)) / slice_size
                    
                    # we have to normalize central x and invert y for yolo format
                    new_x = (centre.coords.xy[0][0] - x1) / slice_size
                    new_y = (y1 - centre.coords.xy[1][0]) / slice_size
                    
                    counter += 1

                    slice_labels.append([box[0], new_x, new_y, new_width, new_height])
            
            # save txt with labels for the current tile
            if len(slice_labels) > 0:
                slice_df = pd.DataFrame(slice_labels, columns=['class', 'x1', 'y1', 'w', 'h'])
                #print(slice_df)
                slice_df.to_csv(slice_labels_path, sep=' ', index=False, header=False, float_format='%.6f')
            
            # if there are no bounding boxes intersect current tile, save this tile to a separate folder 
            if not imsaved:
                sliced = imr[i*slice_size:(i+1)*slice_size, j*slice_size:(j+1)*slice_size]
                sliced_im = Image.fromarray(sliced)
                filename = imname.split('/')[-1]
                #filename = imname
                slice_path = falsepath + filename.replace('.jpg', f'_{i}_{j}.jpg')                

                sliced_im.save(slice_path)
                print('Slice without boxes saved')
                imsaved = True
#     if counter == 100:
#         break


# In[21]:


# Think before running the tiler. took a hell lot of time
# Lets see the distribution now


# In[32]:


import pandas as pd


# In[46]:


df_true = pd.DataFrame(columns = ['name','humans','number_of_humans'])
df_false = pd.DataFrame(columns = ['name','humans','number_of_humans'])


for images_path in glob.glob(root_path+'tiled/ts/*.jpg'):
    label_path = images_path.replace('.jpg','.txt')
    
    with open(label_path,'r') as fp:
        number_of_human = len(fp.readlines())
    df_true = df_true.append({'name':images_path,'humans':1,'number_of_humans':number_of_human},ignore_index=True)
    

df_false['name'] = glob.glob(root_path+'tiled/false/*.jpg')
df_false.humans = 0
df_false.number_of_humans = 0

df = df_true.append(df_false)


# In[47]:


df


# In[48]:


print(df_true.shape,df_false.shape)


# In[49]:


df_true


# In[ ]:




