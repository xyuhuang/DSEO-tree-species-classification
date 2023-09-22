
# import numpy as np
# import matplotlib.pyplot as plt
# from osgeo import gdal
# import pandas as pd
# import os
# import cv2 as cv

# #读写文件
# file_path = 'cropped_data'  # tif文件路径
# npy_path = 'cropped_data_npy' #npy文件路径
# file_species = os.listdir(file_path)   # 读入文件夹
# array_species = os.listdir(npy_path)   # 读入文件夹
# species_num = len(file_species)       # 统计文件夹中的文件个数
# print(file_species) #输出6个目录名


# for i in range(species_num):
#     species_path=( './cropped_data_npy' + '/' + file_species[i])
                
#     try:
#         os.makedirs(species_path)   # 新建以区域名字命名的文件夹
#     except OSError as error:       # 避免重复创建文件夹报错
#         print(error)

# for file in file_species: #对6个目录遍历
#     tif_search = file_path + '/' + file #.tif图像上一级文件路径
#     files_tiff = os.listdir(tif_search)  #以array形式得到6个包含所有tif名字的数组
#     #print(files_tiff) #以array形式输出6个tif名数组

#     for file_t in files_tiff: #对所有tif文件遍历
#         if file_t.endswith('tif'):
#             file_tiff = file_path + '/' + file + '/' + file_t #所有tif的路径
#             dataset = gdal.Open(file_tiff)  # 打开所有tif
#             #print (file_tiff) #输出所有tif的路径

#             tiff_name = os.path.basename(file_tiff) #所有tif的名
#             #print (tiff_name) #输出所有tif的名字
#             tiff_path = os.path.dirname(file_tiff) #所有的tif的路径名
#             #print (tiff_path) #输出cropped_data+species路径名字
#             #output = tiff_path + '/' + os.path.splitext(tiff_name)[0] + '.npy'
#             #num_bands = dataset.RasterCount     # 获取波段数
#             #print(num_bands)
#             tmp_img = dataset.ReadAsArray()      #将数据转为数组
#             #print(tmp_img.shape)
#             img_array = tmp_img.transpose(1, 2, 0)     #由波段、行、列——>行、列、波段
#             #print(img.shape)
#             np.save('cropped_data_npy'+'/'+file+'/'+os.path.splitext(tiff_name)[0], img_array)



# ###### Store Image+Label for ML training
# t = []
# npy_species=[] #6个npy类别文件夹的路径名
# tmp_npy =[]
# datalist =[]
# #training samples
# for i in range(species_num):
#     train_samples= 'cropped_data_npy' + '/'+file_species[i]+'_train_samples' + '.npy'
#     npy_species = 'cropped_data_npy' + '/'+file_species[i]
#     ##print (npy_species)
#     ##print (train_samples)
#     tmp_npy = os.listdir(npy_species)
#     npy_num = len(tmp_npy)
#     for j in range(npy_num):
#         t= np.load(npy_species+'/'+tmp_npy[j])
#         #print(t)
#         datalist.append(t)
#     t_samples_array = np.array(datalist,dtype=object)
#     print(t_samples_array)
#     print(t_samples_array.shape)
#     np.save(train_samples, t_samples_array)
   


import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
import pandas as pd
import os
import cv2 as cv
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#读写文件
file_path = './cropped_data'  # tif文件路径
npy_path = './cropped_data_npy' #npy文件路径
file_species = os.listdir(file_path)   # 读入文件夹
array_species = os.listdir(npy_path)   # 读入文件夹
species_num = len(file_species)       # 统计文件夹中的文件个数
print(file_species) #输出6个目录名


# for i in range(species_num):
#     species_path=( './cropped_data_npy' + '/' + file_species[i])
                
#     try:
#         os.makedirs(species_path)   # 新建以区域名字命名的文件夹
#     except OSError as error:       # 避免重复创建文件夹报错
#         print(error)

# for file in file_species: #对6个目录遍历
#     tif_search = file_path + '/' + file #.tif图像上一级文件路径
#     files_tiff = os.listdir(tif_search)  #以array形式得到6个包含所有tif名字的数组
#     #print(files_tiff) #以array形式输出6个tif名数组

#     for file_t in files_tiff: #对所有tif文件遍历
#         if file_t.endswith('tif'):
#             file_tiff = file_path + '/' + file + '/' + file_t #所有tif的路径
#             dataset = gdal.Open(file_tiff)  # 打开所有tif
#             #print (file_tiff) #输出所有tif的路径

#             tiff_name = os.path.basename(file_tiff) #所有tif的名
#             #print (tiff_name) #输出所有tif的名字
#             tiff_path = os.path.dirname(file_tiff) #所有的tif的路径名
#             #print (tiff_path) #输出cropped_data+species路径名字
#             #output = tiff_path + '/' + os.path.splitext(tiff_name)[0] + '.npy'
#             num_bands = dataset.RasterCount     # 获取波段数
#             #print(num_bands)
#             tmp_img = dataset.ReadAsArray()      #将数据转为数组
#             #print(tmp_img.shape)
            
# ################################ 旋转， 影像增强 #################################################
#             img_array = tmp_img.transpose(1, 2, 0)     #由波段、行、列——>行、列、波段
# #            print(img_array.shape)
#             img_180 = img_array.copy()
#             img_180 = img_array.reshape(int(img_array.size/30),30)#图像像素维度变形为 a*3格式，要保证每个RGB数组不发生改变；
#             img_180 = np.array(img_180[::-1])#进行行逆置
#             img_180 = img_180.reshape(img_array.shape[0],img_array.shape[1],img_array.shape[2])#再对图像进行一次变换，变成 源图像的维度
#             img_180 = img_180.transpose(2, 0, 1)
# #            plt.figure()
# #            plt.imshow(img_array[:,:,0:3])
# #            print(img_array[:,:,10])
# #            plt.figure()
# #            plt.imshow(img_180[:,:,0:3])
# #            print(img_180[:,:,10])
            
#             img_90 = img_array.copy()
#             img_90 = img_array.transpose(1,0,2)#行列转置
#             img_90 = img_90[::-1]
#             img_90 = img_90.transpose(2, 0, 1)
# #            plt.figure()
# #            plt.imshow(img_90[:,:,0:3])
# #            print(img_90.shape)
            
#             img_270 = img_array.copy()
#             img_270 = img_270.transpose(1,0,2)[::-1]
#             img_270 = img_270.reshape(int(img_270.size/30),30)
#             img_270 = np.array(img_270[::-1])
#             #恢复原数组维度，这个需要注意，图像长宽尺寸与原图相反；
#             img_270 = img_270.reshape(img_array.shape[1],img_array.shape[0],img_array.shape[2])
#             img_270 = img_270.transpose(2, 0, 1)
# #            plt.figure()
# #            plt.imshow(img_270[:,:,0:3])
# #            print(img_270[:,:,10])
            
#             # print(img.shape)
            
#             np.save('cropped_data_npy'+'/'+file+'/'+os.path.splitext(tiff_name)[0], tmp_img)
#             np.save('cropped_data_npy'+'/'+file+'/'+'_90_'+os.path.splitext(tiff_name)[0], img_90)
#             np.save('cropped_data_npy'+'/'+file+'/'+'_180_'+os.path.splitext(tiff_name)[0], img_180)
#             np.save('cropped_data_npy'+'/'+file+'/'+'_270_'+os.path.splitext(tiff_name)[0], img_270)



###### Store Image+Label for ML training
t = []
npy_species=[] #6个npy类别文件夹的路径名
tmp_npy =[]
datalist =[]
#t_samples_array_integrate=[]
#training samples
for i in range(species_num):
    train_samples = 'cropped_data_npy' + '/'+file_species[i]+'_train_samples' + '.npy'
    npy_species = 'cropped_data_npy' + '/'+file_species[i]
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


np.save('cropped_data_npy' + '/'+"x_train_merge.npy", X_samples_array_integrate)
np.save('cropped_data_npy' + '/'+"y_train_merge.npy", Y_samples_array)         





# Split train and test data
X_train, X_test, Y_train, Y_test = train_test_split(X_samples_array_integrate, Y_samples_array, test_size=0.2, random_state=42)
print("Training X data shape: ", X_train.shape)
print("Test X data shape: ", X_test.shape)
print("Training Y data shape: ", Y_train.shape)
print("Test Y data shape: ", Y_test.shape)
np.save('cropped_data_npy' + '/'+"x_train.npy", X_train)
np.save('cropped_data_npy' + '/'+"y_train.npy", Y_train)
np.save('cropped_data_npy' + '/'+"x_test.npy", X_test)
np.save('cropped_data_npy' + '/'+"y_test.npy", Y_test)

   
                

            