# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import math

def img_Whitening(img):
    m = np.mean(img)
    v = np.var(img)
    v = math.sqrt(v)
    img_w = (img - m)/v
    min_p = img_w.min()
    max_p = img_w.max()
#    print(min_p)
#    print(max_p)
    img_w = (img_w - min_p)/(max_p - min_p) #将像素归到 0-1 之间
#    print(img_w)
#    img_w = img_w * (img_w > 0) # 将小于0的像素设为0
#    img_w = img_w * (img_w <= 1) +  1 * (img_w > 1) #将大于1的像素设为1
#    plt.figure()
#    plt.imshow(img, plt.cm.gray)
#    plt.figure()
#    plt.imshow(img_w, plt.cm.gray)
    return img_w   


#双线性插值实现
def bilinear_interpolation(img,out_dim):
    src_h,src_w,channels = img.shape
    dst_h,dst_w = out_dim[1],out_dim[0]
    #print("src_h,src_w= ",src_h,src_w)
    #print("dst_h,dst_w= ",dst_h,dst_w)
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    dst_img = np.zeros((dst_h,dst_w,30),dtype=np.float)
    scale_x,scale_y = float(src_w)/dst_w,float(src_h)/dst_h
    for i in range(30): 
        #m = np.img
        #print(m)
        d = pd.DataFrame(data=img[:,:,i])        
        img[:,:,i] = d.fillna(value=0).values
        print(img)
        #print("All Nan filled")

        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                #根据几何中心重合找出目标像素的坐标
                src_x = (dst_x+0.5)*scale_x-1
                src_y = (dst_y+0.5)*scale_y-1
                print(src_x,src_y)
                #找出目标像素最邻近的四个点
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0+1,src_w-1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0+1,src_h-1)
                print(src_x0,src_x1,src_y0,src_y1)
            
                #代入公式计算
                temp0 = (src_x1 - src_x) * img[src_y0,src_x0,i] + (src_x - src_x0) * img[src_y0,src_x1,i]
                temp1 = (src_x1 - src_x) * img[src_y1,src_x0,i] + (src_x - src_x0) * img[src_y1,src_x1,i]
                #print(temp0,temp1)
                a=float((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
                #print(a)
                dst_img[dst_y,dst_x,i] = a
                #print(dst_img[dst_y,dst_x,i])
                #dst_img[dst_y,dst_x,i] = (1-u)*(1-v)*img[y,x,i] + u*(1-v)*img[y+1,x,i] + (1-u)*v*img[y,x+1,i] + u*v*img[y+1,x+1,i]
                print('one tiff:')
    
    img_w = img.copy()
    for i in range(img.shape[2]):
                img_w[:,:,i] = img_Whitening(img[:,:,i]) # 对每一个通道做白化
    img_dst = dst_img.copy()
    for i in range(dst_img.shape[2]):
                img_dst[:,:,i] = img_Whitening(dst_img[:,:,i]) # 对每一个通道做白化
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img_w[:,:,0:3])
    plt.subplot(1, 2, 2)
    plt.title("Upsampled Image")
    plt.imshow(img_dst[:,:,0:3])
    plt.suptitle("Test Data Upsampling")
    print(img_none)
        
    # print(img_none)

                

    return dst_img


#############validation
#############test数据完成上采样
#读写文件
file_path = './TestSet'  # tif文件路径
npy_path = './cropped_data_npy' #npy文件路径
file_species = os.listdir(file_path)   # 读入文件夹
array_species = os.listdir(npy_path)   # 读入文件夹
species_num = len(file_species)       # 统计文件夹中的文件个数
print(file_species) #输出6个目录名

for i in range(species_num):
    species_path=( '/content/cropped_data_test' + '/' + file_species[i])
                
    try:
        os.makedirs(species_path)   # 新建以区域名字命名的文件夹
    except OSError as error:       # 避免重复创建文件夹报错
        print(error)

for file in file_species: #对6个目录遍历
    tif_search = file_path + '/' + file #.tif图像上一级文件路径
    files_tiff = os.listdir(tif_search)  #以array形式得到6个包含所有tif名字的数组
    #print(files_tiff) #以array形式输出6个tif名数组

    for file_t in files_tiff: #对所有tif文件遍历
        print(file_t)
        if file_t.endswith('tif'):
            file_tiff = file_path + '/' + file + '/' + file_t #所有tif的路径
            dataset = gdal.Open(file_tiff)  # 打开所有tif
            #print (file_tiff) #输出所有tif的路径

            tiff_name = os.path.basename(file_tiff) #所有tif的名
            print (tiff_name) #输出所有tif的名字
            tiff_path = os.path.dirname(file_tiff) #所有的tif的路径名
            #print (tiff_path) #输出cropped_data+species路径名字
            #output = tiff_path + '/' + os.path.splitext(tiff_name)[0] + '.npy'
            num_bands = dataset.RasterCount     # 获取波段数
            #print(num_bands)
            tmp_img = dataset.ReadAsArray()      #将数据转为数组
            #print(tmp_img.shape)

            img_array = tmp_img.transpose(1, 2, 0)     #由波段、行、列——>行、列、波段


            #data = np.arange(17280).reshape(3,3,3)
            #print 'Original:\n', data
            #print 'Zoomed by 2x gives an array of shape:', ndimage.zoom(data, 2).shape
            #data_up = ndimage.zoom(tmp_img, (1, 2, 2))

            
            dst_img = bilinear_interpolation(img_array,(32,32))
            #print(dst_img)
            a_name = np.save('./upsample_data'+'/'+file+'/'+os.path.splitext(tiff_name)[0], dst_img)
            
            #print(img_none)
            print(a_name)
            print(dst_img.shape)



#合并npy
###### Store Image+Label for ML training
t = []
npy_species=[] #6个npy类别文件夹的路径名
tmp_npy =[]
datalist =[]
#t_samples_array_integrate=[]
#training samples
for i in range(species_num):
    train_samples = 'upsample_data' + '/'+file_species[i]+'_train_samples' + '.npy'
    npy_species = 'upsample_data' + '/'+file_species[i]
    tmp_npy = os.listdir(npy_species)
    npy_num = len(tmp_npy)
    #print(npy_num)
    if npy_num==0:
        continue
    for j in range(npy_num):
        t= np.load(npy_species+'/'+tmp_npy[j])
        #print(t)
        datalist.append(t)
        #print(j)
        #print(np.array(datalist,dtype=object))
    t_samples_array = np.array(datalist,dtype=object)
    #t_samples_array_integrate = np.concatenate((t_samples_array_integrate, t_samples_array))
    datalist=[]
    if i==0:
        X_samples_array_integrate = t_samples_array
        Y_samples_array = np.zeros(npy_num)+i
    else:
        Y_samples_array = np.concatenate((Y_samples_array, np.zeros(npy_num)+i))
        X_samples_array_integrate = np.concatenate((X_samples_array_integrate, t_samples_array))
    
    #print(t_samples_array.shape)
    #print(v_samples_array.shape)
    #print(t_samples_array)
    #print(np.array(datalist,dtype=object).shape)
    #print(t_samples_array.shape)
    #print(t_classes_array.shape)
    #t_samples_array = []
    np.save(train_samples, t_samples_array)
#t_train_samples

# fill Nan using the previous line
number_samples = X_samples_array_integrate.shape[0]
m = np.reshape(X_samples_array_integrate,(number_samples,30*32*32))
d = pd.DataFrame(data=m)        
m = d.fillna(value=0).values
X_samples_array_integrate = np.reshape(m,(number_samples,30,32,32))
print("All Nan filled")

np.save('upsample_data' + '/'+"x_test_merge.npy", X_samples_array_integrate)
np.save('upsample_data' + '/'+"y_test_merge.npy", Y_samples_array)         

  
# Split train and test data
X_train, X_test, Y_train, Y_test = train_test_split(X_samples_array_integrate, Y_samples_array, test_size=0.2, random_state=42)
print("Training X data shape: ", X_train.shape)
print("Test X data shape: ", X_test.shape)
print("Training Y data shape: ", Y_train.shape)
print("Test Y data shape: ", Y_test.shape)
np.save('upsample_data' + '/'+"x_train.npy", X_train)
np.save('upsample_data' + '/'+"y_train.npy", Y_train)
np.save('upsample_data' + '/'+"x_test.npy", X_test)
np.save('upsample_data' + '/'+"y_test.npy", Y_test)                
    






