from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options

import sys
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

#Need to install Selenium driver, installation and usage depends on the OS (this is for linux)
def move_tif_files(dest_dir, wells_dir):
    for filename in os.listdir(dest_dir):
        if filename.endswith(".tif"):
            image_path = os.path.join(dest_dir, filename)
            destination_path = os.path.join(wells_dir, filename)
            shutil.move(image_path, destination_path)

directory = 'nm_wells'
os.chdir(directory)
wells_dir = "nm_all_wells"
source_dir = "Downloads/test_" + str(int(float(sys.argv[1]))) +  '/'
dest_dir = "nm_wells/files_" + str(int(float(sys.argv[1]))) +  '/'
if not os.path.exists(wells_dir):
    os.makedirs(wells_dir)

if not os.path.exists(source_dir):
    os.makedirs(source_dir)
    print("created dir: ", source_dir)

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)


df = pd.read_pickle('positive_metadata_to_process.pkl') 
ents = list(dict.fromkeys(df['entityId']))
ents = list(df['entityId'].value_counts().index) 
df['displayId'] = df['displayId'].str.lower()

output_prefix = 'tile'
tile_size = (512,512)
tile_height, tile_width = tile_size
ext = ['tif']    
wbuffer = 50
datasetID = '5e83a340bf820c39'
error_i_list = []

tiles_dir = os.path.join(dest_dir, str(tile_size[0]))
df['num_bands'] = 'Unknown'
df['pixel_resolution'] = 'Unknown'
df['west'] = 'Unknown'
df['east'] = 'Unknown'
df['south'] = 'Unknown'
df['north'] = 'Unknown'
df['date'] = 'Unknown'
df['height'] = 'Unknown'
df['width'] = 'Unknown'
df['tile_name'] = [[] for _ in range(len(df))]
df['input_point'] = [[] for _ in range(len(df))]

#this might depend on the Selenium version and OS
options = Options()
options.add_argument("--headless")
#profile = webdriver.FirefoxProfile()
options.set_preference("browser.download.folderList", 2)
#profile.set_preference("browser.download.folderList", 2)
options.set_preference("browser.download.dir", source_dir)
#profile.set_preference("browser.download.dir", source_dir)
#driver = webdriver.Firefox(options=options, firefox_profile=profile)
#options.profile = profile
service = Service("geckodriver")
driver = webdriver.Firefox(options=options, service=service)

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
            driver.find_element(By.ID, "loginButton").submit()
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
            transformer = Transformer.from_crs('EPSG:4326', src.crs, always_xy=True)
            transform = src.transform
            left, bottom, right, top = src.bounds
            left, bottom, right, top = transform_bounds(src.crs, 'EPSG:4326', left, bottom, right, top)
            df.loc[df['displayId'] == filename_string.split(".")[0], 'num_bands'] = int(num_bands)
            df.loc[df['displayId'] == filename_string.split(".")[0], 'pixel_resolution'] = transform[0]
            df.loc[df['displayId'] == filename_string.split(".")[0], 'west'] = left
            df.loc[df['displayId'] == filename_string.split(".")[0], 'east'] = right
            df.loc[df['displayId'] == filename_string.split(".")[0], 'south'] = bottom
            df.loc[df['displayId'] == filename_string.split(".")[0], 'north'] = top
            df.loc[df['displayId'] == filename_string.split(".")[0], 'height'] = int(height)
            df.loc[df['displayId'] == filename_string.split(".")[0], 'width'] = int(width)
            df.loc[df['displayId'] == filename_string.split(".")[0], 'date'] = int(filename_string[filename_string.rfind('/') + 1:filename_string.rfind('.')][-8:-4])
        if len(np.unique(df.loc[df['displayId'] == filename_string.split(".")[0], 'entityId'])) == 1:
            tile_tiff(dest_dir + filename_string, ents[i], tile_size)
            print(i, ents[i])
            move_tif_files(tiles_dir, wells_dir)
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
    if (i+1) % 5 == 0:   
        print("df: all_df.pkl is updated") 
        df.to_pickle(directory + "/all_df_" + str(int(float(sys.argv[1]))) + ".pkl")     
        pd.DataFrame(error_i_list).to_pickle(directory + "/all_error_i_list_" + str(int(float(sys.argv[1]))) + ".pkl")
print("error list")        
print(error_i_list)
df.to_pickle(directory + "/allchunk_df_" + str(int(float(sys.argv[1]))) + ".pkl") #part of the positive_df.pkl 
driver.quit()
