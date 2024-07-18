#get images from random locations and complete neccessary items in metadata
#!/usr/bin/env python3
#SET DIR where M2M-api is
#may require download proxies 

from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options

import time 
import rasterio
from rasterio.windows import Window
from rasterio import Affine
from rasterio.warp import transform_bounds
from pyproj import Transformer
import folium 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image, ImageDraw
from geopy import distance
from geopy.point import Point
#from api import M2M
import urllib.request
import warnings
import os.path as osp
import math
import requests
import glob
from rasterio.plot import show
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject, Resampling
from dateutil.parser import parse
import shutil
from datetime import datetime
import pickle
import re
import zipfile
import random
import subprocess
warnings.filterwarnings('ignore')

from extract_functions import closest_divisor, draw_bounding_box, move_image, extract_resolution, save_to_pickle, load_from_pickle, get_substrings_between_substrings, find_items_in_list_of_dicts, order_list_of_dicts, find_characters_after_substring, find_characters_before_substring, find_value_in_list_of_lists, extract_dict_items, add_element, is_coords_inside_bounding_box, get_tile, bbox, calculate_tile, tile_tiff, unzip_and_move

Image.MAX_IMAGE_PIXELS = None


directory = 'nm_wells_random'
os.chdir(directory)
wells_dir = "nm_all_wells_random"
source_dir = "Downloads/test_random_" + str(int(float(sys.argv[1]))) +  '/'
dest_dir = "nm_wells_random/files_random_" + str(int(float(sys.argv[1]))) +  '/'


res_df = pd.read_pickle('/home/exx/Documents/owells/negative_metadata_to_process.pkl')
df = res_df[~res_df['displayId'].isin(['Unknown'])]
ents = list(df['entityId'].value_counts().index) 
df['displayId'] = df['displayId'].str.lower()

output_prefix = 'tile'
tile_size = (512,512)
tile_height, tile_width = tile_size
ext = ['tif']    
wbuffer = 50
datasetID = '5e83a340bf820c39'
error_i_list = []

df['num_bands'] = 'Unknown'
df['pixel_resolution_x'] = 'Unknown'
df['pixel_resolution_y'] = 'Unknown'
df['west'] = 'Unknown'
df['east'] = 'Unknown'
df['south'] = 'Unknown'
df['north'] = 'Unknown'
df['date'] = 'Unknown'
df['height'] = 'Unknown'
df['width'] = 'Unknown'
df['tile_name'] = [[] for _ in range(len(df))]
df['input_point'] = [[] for _ in range(len(df))]


driver = webdriver.Firefox()
os.chdir(dest_dir)


for i in range(int(float(sys.argv[1])), int(float(sys.argv[2]))):
    print(i)
    url = "https://earthexplorer.usgs.gov/download/options/naip/" + str(ents[i]) + "/"
    #print(url)
    try:
        driver.get(url)
        if i == int(float(sys.argv[1])):
            username_field = driver.find_element(By.NAME, 'username').send_keys('your_username')
            password_field = driver.find_element(By.NAME, 'password').send_keys('your_password')
        WebDriverWait(driver, 60).until(EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Download')]"))).click()
        while not any(fname.endswith('.part') for fname in os.listdir(source_dir)):
            time.sleep(3)
        print("entityId: ", i, ents[i])
        while any(fname.endswith('.part') for fname in os.listdir(source_dir)):
            time.sleep(3) 
        for filename in os.listdir(source_dir):
            if filename.endswith(".ZIP"):
                source_path = os.path.join(source_dir, filename)
                dest_path = os.path.join(dest_dir, filename)
                shutil.move(source_path, dest_path)
                with zipfile.ZipFile(dest_path, 'r') as zip_ref:
                    zip_ref.extractall(dest_dir)
                os.remove(dest_path)
        all_files = os.listdir(dest_dir)
        files = [f for f in all_files if f.endswith('.tif')]
        filename_string = files[0]
        temp_tiles = []
        with rasterio.open(dest_dir + filename_string) as src:
            metadata = src.meta
            height, width = src.height, src.width
            num_bands = src.count
            crs = src.crs
            transformer = Transformer.from_crs('EPSG:4326', src.crs, always_xy=True)#transformer = Transformer.from_crs("EPSG:26917", "EPSG:4326")
            transform = src.transform
            left, bottom, right, top = src.bounds
            left, bottom, right, top = transform_bounds(src.crs, 'EPSG:4326', left, bottom, right, top)
            df.loc[df['displayId'] == filename_string.split(".")[0], 'num_bands'] = int(num_bands)
            df.loc[df['displayId'] == filename_string.split(".")[0], 'pixel_resolution_x'] = transform[0]
            df.loc[df['displayId'] == filename_string.split(".")[0], 'pixel_resolution_y'] = -transform[4]
            df.loc[df['displayId'] == filename_string.split(".")[0], 'west'] = left
            df.loc[df['displayId'] == filename_string.split(".")[0], 'east'] = right
            df.loc[df['displayId'] == filename_string.split(".")[0], 'south'] = bottom
            df.loc[df['displayId'] == filename_string.split(".")[0], 'north'] = top
            df.loc[df['displayId'] == filename_string.split(".")[0], 'height'] = int(height)
            df.loc[df['displayId'] == filename_string.split(".")[0], 'width'] = int(width)
            df.loc[df['displayId'] == filename_string.split(".")[0], 'date'] = int(filename_string[filename_string.rfind('/') + 1:filename_string.rfind('.')][-8:-4])
        if len(np.unique(df.loc[df['displayId'] == filename_string.split(".")[0], 'entityId'])) == 1:
            tile_tiff(dest_dir + filename_string, ents[i], tile_size)
            for index, row in df[df['displayId'] == filename_string.split(".")[0]].iterrows():
                xx, yy = transformer.transform(row['long'], row['lat'])
                plat,plong=src.index(xx,yy)
                if plat < height and plat > 0 and plong > 0 and plong < width:
                    input_point = calculate_tile(plat, plong, (width, height), tile_size, i)
                    lat_tile = closest_divisor(plat, tile_height)
                    long_tile = closest_divisor(plong, tile_height)
                    target_tile_name = ents[i] + '_' + str(int(lat_tile)) + "_" + str(int(long_tile))
                    row['tile_name'].append(target_tile_name)
                    row['input_point'].append(input_point)
                    temp_tiles.append(target_tile_name)
                unique_tiles = np.unique(temp_tiles)
            for j in range(len(unique_tiles)):
                cur_df = df[df['tile_name'].apply(lambda x: unique_tiles[j] in x)]
                input_points_list = list(cur_df['input_point'])
                well_mask = np.zeros((tile_size[0], tile_size[1], 3), dtype = np.uint8) 
                cv2.imwrite(masks_dir + '/negmask_' + unique_tiles[j] + '.png', well_mask)
                image_path = dest_dir + str(tile_size[0]) + "/" + unique_tiles[j] + ".tif"
                image = cv2.imread(image_path)
                for t in range(len(input_points_list)):
                    x, y, width, height = input_points_list[t][0][0], input_points_list[t][0][1], wbuffer, wbuffer
                    top_left = (x - width, y - height)
                    bottom_right = (x + width, y + height)
                    cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 2) 
                shutil.move(image_path, wells_dir + '/negwell_' + unique_tiles[j] + ".tif")
            for filename in os.listdir(dest_dir):
                file_path = os.path.join(dest_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)  
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  
        else:
            print("Not unique entities ", i, ents[i])
            error_i_list.append(i)
    except Exception as e:
        print(f"Error on iteration {i}: {e}")
    if (i+1) % 10 == 0:   
        print("df: rand_df.pkl is updated") 
        df.to_pickle(directory + "/all_df_random_" + str(int(float(sys.argv[1]))) + ".pkl")     
        pd.DataFrame(error_i_list).to_pickle(directory + "/error_i_list.pkl")
print("error list")        
print(error_i_list)
df.to_pickle(directory + "/allchunk_df_random_" + str(int(float(sys.argv[1]))) + ".pkl") #part of the negative_df.pkl 
driver.quit()
