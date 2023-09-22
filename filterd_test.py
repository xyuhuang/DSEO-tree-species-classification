# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 01:18:08 2023

@author: 93982
"""

#####################
#在这个 .py文件目录下手动添加一个文件夹 “cropped_data”
#图像数据data目录放在.py文件目录同一级下
#data文件夹下目录名为下载图像的index，即图像区域 如23，24
#####################

import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
import pandas as pd
import os
from pyproj import Transformer, CRS
from PIL import Image

#读写文件
file_path = './data'  # tif文件路径
file_tree = 'File1_output_3000.csv' #五个树种
files = os.listdir(file_path)   # 读入文件夹
num = len(files)       # 统计文件夹中的文件个数
print(num)

#land cover
file_lc = './data_lc/E000N60_PROBAV_LC100_global_v3.0.1_2019-nrt_Tree-CoverFraction-layer_EPSG-4326.tif'

for file in files:
    tif_search = file_path + '/' + file #.tif图像上一级文件路径
    files_tiff = os.listdir(tif_search)
    #print(files_tiff)
      
    for file_t in files_tiff:
        if file_t.endswith('tif'):
            file_tiff = file_path + '/' + file + '/' + file_t
            dataset = gdal.Open(file_tiff)  # 打开tif
            #print (file_tiff)
 
            tiff_name = os.path.basename(file_tiff)
            tiff_path = os.path.dirname(file_tiff)
            output = tiff_path + '/' + os.path.splitext(tiff_name)[0] + '.csv'
 	       
            #读tif坐标
            geo_information = dataset.GetGeoTransform()
            col = dataset.RasterXSize
            row = dataset.RasterYSize
    
            #图像右下角经纬度
            lon = geo_information[0] + col * geo_information[1] + row * geo_information[2]
            lat = geo_information[3] + col * geo_information[4] + row * geo_information[5]
    
    
            #读点坐标
            data = pd.read_csv(file_tree,sep=',',header = 0)
            df = data[data['X'].between(geo_information[0], lon, inclusive=True)]
            tree_points = df[df['Y'].between(lat, geo_information[3], inclusive=True)]
    
            tree_points.to_csv(output, index = False)
      
    
            #geo_information(0):左上像素左上角的x坐标。
            #geo_information(1):w - e像素分辨率 / 像素宽度。
            #geo_information(2):行旋转（通常为零）。
            #geo_information(3):左上像素左上角的y坐标。
            #geo_information(4):列旋转（通常为零）。
            #geo_information(5):n - s像素分辨率 / 像素高度（北半球上图像为负值）
  
    
####裁剪     
def crop_file():  
    for file in files:
        tif_search = file_path + '/' + file #.tif图像上一级文件路径
        files_tiff = os.listdir(tif_search)
        #print(files_tiff)
      
        for file_t in files_tiff:
            if file_t.endswith('tif'):
                file_tiff = file_path + '/' + file + '/' + file_t
                dataset = gdal.Open(file_tiff)  # 打开tif
                #print (file_tiff)
     
                tiff_name = os.path.basename(file_tiff) #tif图像名字
                tiff_path = os.path.dirname(file_tiff) #eg.data/23
                      
   
                #print (file_tiff)
                area_name = os.path.basename(os.path.dirname(file_tiff))  #文件路径 eg.34，35
             	
                #创建树种文件夹
                tree = pd.read_csv(file_tree,sep=',',header = 0)
                tree_species = list(set(tree['SPECIES NAME']))
                print(tree_species)
                
                for i in range(5): #五个树种
                    species_path=( './cropped_data' + '/' + tree_species[i])
                
                    try:
                      os.makedirs(species_path)   # 新建以区域名字命名的文件夹
                    except OSError as error:       # 避免重复创建文件夹报错
                      print(error)
                  
                
                #读tif坐标
                geo_information = dataset.GetGeoTransform()
                col = dataset.RasterXSize
                row = dataset.RasterYSize
                print('行列：',col,row)

                
                for j in range(len(open(csvfile).readlines())-1): #每个csv文件每一行
                    win=( ulx[j], uly[j], lrx[j], lry[j] ) #裁剪区域
                    output_tif = './cropped_data' + '/' + point_data['SPECIES NAME'] + '/' + area_name + '_' + str(j) + '_' + tiff_name

                    lc_tif = './cropped_data' + '/' + point_data['SPECIES NAME'] + '/' + area_name + '_' + str(j) + '_lc_' + tiff_name 
                    print(output_tif)                                    

                    
                    #NDVI过滤
                    small_tif = gdal.Open(output_tif[j])
                    im_bands = small_tif.RasterCount #波段数              
                    
                    scols = small_tif.RasterXSize  # 列数
                    srows = small_tif.RasterYSize  # 行数
                 
                    band8 = small_tif.GetRasterBand(7).ReadAsArray(0, 0, scols, srows) #nir
                    band4 = small_tif.GetRasterBand(3).ReadAsArray(0, 0, scols, srows) #red
                    
                    molecule = band8 - band4
                    denominator = band8 + band4
                    del small_tif
                    ndvi = molecule / denominator #每个像素
                    NDVI = np.mean(ndvi) #整幅图像
                    print(NDVI)
                    
                    print(type(NDVI))
                    print(np.shape(NDVI))
                    #print(ndvi[j-1])

                    #Land Cover过滤
                    lc_filter_tif = gdal.Open(lc_tif[j])
                    scols = lc_filter_tif.RasterXSize  # 列数
                    srows = lc_filter_tif.RasterYSize  # 行数
                    lc_band = lc_filter_tif.GetRasterBand(1).ReadAsArray(0, 0, scols, srows) #获取影像灰度值
                    lc_mean = np.mean(lc_band)  
                    del lc_filter_tif
                    print(lc_mean)

                    if NDVI < 0.5 or lc_mean < 70: #lc像素值0-95，白色为forest
                        os.remove(output_tif[j])
                    os.remove(lc_tif[j]) #删除生成的土地分类影像
                   
crop_file()   
    
    
    
    
    
    
    
    
    
    











