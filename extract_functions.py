import rasterio
from rasterio.windows import Window
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
import pandas as pd
from pyproj import Transformer
import rasterio
from rasterio import Affine
import numpy as np
import glob
import os
from PIL import Image, ImageDraw
import pickle


def unzip_and_move(zip_path):
    dir_name = os.path.dirname(zip_path)
    base_name = os.path.basename(zip_path).replace('.ZIP', '')
    zip_files = glob.glob(os.path.join(zip_path, "*.ZIP"))
    with zipfile.ZipFile(zip_files[0], 'r') as zip_ref:
        zip_ref.extractall(dir_name + "/" + base_name)
    os.remove(zip_files[0])


def tile_tiff(path, output_prefix, tile_size):
    directory = "./" + str(tile_size[0])
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    else:
        print(f"Directory '{directory}' already exists.")
    img = Image.open(path)
    img_width, img_height = img.size
    tile_width, tile_height = tile_size
    # Split the image into separate bands
    bands = img.split()
    # Ensure there are at least three bands
    if len(bands) < 3:
        raise ValueError("Image does not have RGB bands")
    # Extract the RGB bands (assuming RGB order)
    red_band = bands[0]
    green_band = bands[1]
    blue_band = bands[2]
    # Create a new RGB image with only the RGB bands
    rgb_image = Image.merge("RGB", (red_band, green_band, blue_band))
    for y in range(0, img_height, tile_height):
        for x in range(0, img_width, tile_width):
            tile = rgb_image.crop((x, y, x + tile_width, y + tile_height))
            # Get the output tile filename based on row and column indices
            tile_filename = f"{output_prefix}_{y}_{x}.tif"
            # Save the tile as a TIFF file
            tile.save("./" + str(tile_size[0]) + "/" + tile_filename, format='TIFF')
    img.close()
    
    

def closest_divisor(n, tile_height): #assuming tile_height = tile_width
    if n <= tile_height:
        return 0
    remainder = (n - 1) % tile_height
    return n - 1 - remainder

def draw_bounding_box(image_path, bbox, output_path):
    image = Image.open(image_path)
    drawn_image = image.copy()
    draw = ImageDraw.Draw(drawn_image)
    draw.rectangle(bbox, outline="red", width=2)
    drawn_image.save(output_path)

def move_image(source_path, destination_path):
    try:
        shutil.move(source_path, destination_path)
        print("File moved successfully.")
    except FileNotFoundError:
        print("Source file not found.")
        
        

def extract_resolution(url):
    pattern1 = r'0x\d+m'
    pattern2 = r'1x\d+m'
    matches1 = re.findall(pattern1, url)
    matches2 = re.findall(pattern2, url)
    if matches1:
        resolution = matches1[0][2:-1]
        return f"0x{resolution}"
    elif matches2:
        resolution = matches2[0][2:-1]
        return f"1x{resolution}"
    else:
        return None
    
    
def save_to_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
        
def load_from_pickle(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

    
def get_substrings_between_substrings(original_list, start_substring, end_substring):
    result_list = []
    for original_string in original_list:
        start_index = original_string.find(start_substring)
        end_index = original_string.find(end_substring, start_index + len(start_substring))
        if start_index != -1 and end_index != -1:
            substring = original_string[start_index + len(start_substring):end_index]
            result_list.append(substring)
    return result_list

def find_items_in_list_of_dicts(lst, key, items):
    result_indices = []
    for item in items:
        found_index = None
        for index, dictionary in enumerate(lst):
            if key in dictionary and dictionary[key] == item:
                found_index = index
                break
        result_indices.append(found_index)
    return result_indices

def order_list_of_dicts(lst, indices):
    ordered_list = [lst[i] if i is not None else {} for i in indices]
    return ordered_list


def find_characters_after_substring(string_list, substring, char_num):
    res = []
    for string in string_list:
        index = string.find(substring)
        if index != -1 and index + len(substring) + char_num <= len(string):
            res.append(string[index + len(substring):index + len(substring) + char_num])
    return res

def find_characters_before_substring(string_list, substring, char_num):
    res = []
    for string in string_list:
        index = string.find(substring)  # Find the starting index of the substring in the string
        characters = string[index - char_num:index]  # Extract the three characters before the substring
        res.append(characters)
    return res

def find_value_in_list_of_lists(data, key):
    res = []
    for sublist in data:
        for dictionary in sublist:
            if key in dictionary:
                res.append(dictionary[key])
    return res

def extract_dict_items(dictionary, indices_to_extract):
    extracted_items = {}
    for index in indices_to_extract:
        keys = list(dictionary.keys())
        if index < len(keys):
            key = keys[index]
            extracted_items[key] = dictionary[key]
    return extracted_items

def add_element(dict, key, value):
    if key not in dict:
        dict[key] = []
    dict[key].append(value)
    
def is_coords_inside_bounding_box(target_lat, target_lon, lower_left_lat, lower_left_lon, upper_right_lat, upper_right_lon):
      if lower_left_lat <= target_lat <= upper_right_lat and lower_left_lon <= target_lon <= upper_right_lon:
          return True
      return False
  


def get_tile(imdir, output_prefix, path, tile_size, latitude,longitude):
    img = rasterio.open(path)
    with rasterio.open(path) as src:
        transformer = Transformer.from_crs('EPSG:4326', src.crs, always_xy=True)
        xx, yy = transformer.transform(longitude, latitude)
        row, col = src.index(xx, yy)
    img_width, img_height = img.shape[0], img.shape[1]
    image_resolution = (img_width, img_height)
    plat = row
    plong = col
    tile_name, input_point = calculate_tile(plat, plong, image_resolution, tile_size, output_prefix)
    if tile_name is not None:
        print(f"The desired coordinate ({latitude}, {longitude}) corresponds to tile {tile_name}.")
        #tile_image = np.array(tiles[tile_name])
        plt.imshow(Image.open(imdir + "_" + str(output_prefix) + "/" + str(tile_size[0]) + "/" + tile_name + ".tif"))
        plt.axis('on')
        plt.show()
    else:
        print("No tile corresponds to the desired coordinate.")
    return tile_name, input_point
    
    
     
def bbox(center_latitude, center_longitude, distance_meters):
    center_point = Point(center_latitude, center_longitude)
    lower_left_point = distance.distance(meters=distance_meters).destination(center_point, 225)
    upper_right_point = distance.distance(meters=distance_meters).destination(center_point, 45)

    ll_lat, ll_long = lower_left_point.latitude, lower_left_point.longitude
    ur_lat, ur_long = upper_right_point.latitude, upper_right_point.longitude

    return ll_lat, ll_long, ur_lat, ur_long
   

def calculate_tile(pixel_latitude, pixel_longitude, f_image_resolution, tile_size, output_prefix):
    image_width, image_height = f_image_resolution
    tile_width, tile_height = tile_size
    for y in range(0, image_height, tile_height):
        for x in range(0, image_width, tile_width):
            if y  <= pixel_latitude <= y + tile_height and x <= pixel_longitude <= x + tile_width:
                input_point = np.array([pixel_longitude-x, pixel_latitude-y])
    return input_point #tile_name, input_point

