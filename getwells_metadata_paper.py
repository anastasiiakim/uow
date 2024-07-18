#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Get metadata for positive examples: entityId for the best resolution for each lat&long location
Produced metadata pickle file is used as input for another script (selenium one) to update with well location
"""

#run from directory where folder https://github.com/Fergui/m2m-api is located
import numpy as np
import pandas as pd
import random
import os
from geopy import distance
from geopy.point import Point
from myapi import M2M #this is https://github.com/Fergui/m2m-api
import warnings

warnings.filterwarnings('ignore')
os.environ['http_proxy'] = "http://proxyout.lanl.gov:8080"
os.environ['https_proxy'] = "http://proxyout.lanl.gov:8080" 
os.environ["ftp_proxy"] = "http://proxyout.lanl.gov:8080"
os.environ["HTTP_PROXY"] = "http://proxyout.lanl.gov:8080"
os.environ["HTTPS_PROXY"] = "http://proxyout.lanl.gov:8080"
os.environ["FTP_PROXY"] = "http://proxyout.lanl.gov:8080"

m2m = M2M("your_username", "your_password") #Obtain from USGS Machine API (username, password)

import os
import warnings
warnings.filterwarnings('ignore')

state_abbreviations = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}



    
def get_substrings_between_substrings(original_list, start_substring, end_substring):
    result_list = []
    
    for original_string in original_list:
        start_index = original_string.find(start_substring)
        end_index = original_string.find(end_substring, start_index + len(start_substring))
        #print(original_string, start_index, end_index)

        if start_index != -1 and end_index != -1:
            substring = original_string[start_index + len(start_substring):end_index]
            result_list.append(substring)

    return result_list



def find_characters_after_substring(string_list, substring, char_num):
    res = []
    for string in string_list:
        index = string.find(substring)
        if index != -1 and index + len(substring) + char_num <= len(string):
            res.append(string[index + len(substring):index + len(substring) + char_num])
    return res


def find_value_in_list_of_lists(data, key):
    res = []
    for sublist in data:
        for dictionary in sublist:
            if key in dictionary:
                res.append(dictionary[key])
    return res



def bbox(center_latitude, center_longitude):
    center_point = Point(center_latitude, center_longitude)
    lower_left_point = distance.distance(meters=distance_meters).destination(center_point, 225)
    upper_right_point = distance.distance(meters=distance_meters).destination(center_point, 45)

    ll_lat, ll_long = lower_left_point.latitude, lower_left_point.longitude
    ur_lat, ur_long = upper_right_point.latitude, upper_right_point.longitude

    return ll_lat, ll_long, ur_lat, ur_long


# 2022 -2021 = 82k wells; read files from Mary Kang's paper: https://iopscience.iop.org/article/10.1088/1748-9326/acdae7/meta
wells_2021 = pd.ExcelFile("~/Documents/m2m-api/es2c03268_si_002.xlsx") #download this file from the Documented Orphaned Oil and Gas Wells Across the United States paper by Boutot et. al, 2022 (link at the end of the paper)
wells_2021 = wells_2021.parse('2021 Orphan Well Dataset', header=0)
wells_2022 = pd.ExcelFile("~/Documents/m2m-api/es2c03268_si_003.xlsx") #download this file from the Documented Orphaned Oil and Gas Wells Across the United States paper by Boutot et. al, 2022
wells_2022 = wells_2022.parse('2022 Orphan Well Dataset', header=0)
wells_2022['index'] = wells_2022.index
wells_2021[wells_2021.duplicated(subset=['Latitude', 'Longitude'], keep=False)]
wells_2022[wells_2022.duplicated(subset=['Latitude', 'Longitude'], keep=False)]
wells_2021 = wells_2021.drop_duplicates(subset=['Latitude', 'Longitude'])
wells_2022 = wells_2022.drop_duplicates(subset=['Latitude', 'Longitude'])

merged_df = pd.merge(wells_2021, wells_2022, how='outer', indicator=True)
merged_df[merged_df['_merge'] == 'left_only'].shape
merged_df[merged_df['_merge'] == 'right_only'].shape
merged_df[merged_df['_merge'] == 'both'].shape
df = merged_df[merged_df['_merge'] == 'right_only']


distance_meters = 50 #radius, diagonal
col_lat = df.columns.get_loc('Latitude')
col_long = df.columns.get_loc('Longitude')
nmdf = df.copy()

#get bbox around locations
result_df = pd.DataFrame(columns=['index', 'State', 'lat', 'long', 'll_lat', 'll_long', 'ur_lat', 'ur_long', 'accuracy'])
for idx, row in nmdf.iterrows():
    ll_lat, ll_long, ur_lat, ur_long = bbox(row['Latitude'], row['Longitude'])
    intersects = ((result_df['ll_lat'] <= row['Latitude']) & (result_df['ur_lat'] >= row['Latitude']) &
                  (result_df['ll_long'] <= row['Longitude']) & (result_df['ur_long'] >= row['Longitude'])).any()
    if not intersects:
        result_df = pd.concat([result_df, pd.DataFrame([{'index': idx, 'state': row['State'], 'lat': row['Latitude'], 'long': row['Longitude'], 'll_lat': ll_lat, 'll_long': ll_long, 'ur_lat': ur_lat, 'ur_long': ur_long, 'accuracy': row['Accuracy Note']}])], ignore_index=True)
     
#result_df.to_pickle("result_df_boxes.pkl")
#result_df = pd.read_pickle("result_df_boxes.pkl") #load pre-saved bbox nmdf 
nmdf = result_df
nmdf['index'] = nmdf['index'].astype('int')
nmdf['numBands'] = 'NaN'
nmdf['resolution'] = 'NaN'
nmdf['browsePathlink'] = 'NaN'
nmdf['entityId'] = 'NaN'
nmdf['displayId'] = 'NaN'
nmdf['scenesDate'] = 'NaN'
if (nmdf['accuracy'].isna().sum() != nmdf.shape[0]):
    print("Warning. Some wells locations are not accurate!")



# #will disconnect every 100 requests or so
# Need to request free access on USGS machine API website to get login credentials
m2m = M2M("your_username", "your_password")
label='m2m-api_download'
labels = [label]
datasetName = 'naip'


for idx, row in nmdf.iterrows():
    print("idx ", idx)
    if idx%100 == 0:
        print(idx)
        m2m = M2M("your_username", "your_password")

    params = {
        "datasetName": "naip",
        "startDate": "2015-01-01", #cnir colors for images from 2015
        "endDate": "2023-06-01",
        "boundingBox": (row['ll_long'], row['ur_long'], row['ll_lat'], row['ur_lat']) 
        #"maxCC": 50,
        #"includeUnknownCC": False
        #"boundingBox": (lower_left_longitude,  upper_right_longitude, lower_left_latitude, upper_right_latitude),
        #"maxResults": maxresults
    }
    scenes = m2m.searchScenes(**params)
    if scenes['totalHits'] == 0:
        print(idx, nmdf.iloc[idx, 0])
    print("hits: ", scenes['totalHits'])
    if scenes['totalHits'] == 0:
        print("Error, no recent images found. Change the date.", idx, row)
    else:    
        scenes_list = [scene['browse'] for scene in scenes['results']]
        browsePathlink = find_value_in_list_of_lists(scenes_list, 'browsePath')
        scenes_dates = [scene['temporalCoverage'] for scene in scenes['results']]
        pixel_resolution = find_characters_after_substring(browsePathlink, 'naip_', 4) 
        entityIds = [scene['entityId'] for scene in scenes['results']]
        displayIds = [scene['displayId'] for scene in scenes['results']]
        nmdf.at[idx, 'entityId'] = entityIds#downloads
        nmdf.at[idx, 'displayId'] = displayIds
        nmdf.at[idx, 'scenesDate'] = scenes_dates
        nmdf.at[idx, 'resolution'] = pixel_resolution
        nmdf.at[idx, 'browsePathlink'] = browsePathlink#len(nmdf['url'][idx])

alldf = nmdf
skip_indices = alldf[alldf['entityId'] == 'NaN'].index.tolist()
alldf['orderedId'] = 'NaN'
alldf['selectedIdFromList'] = 'NaN'
alldf['year'] = 'NaN'
for idx, row in alldf.iterrows():
    if idx in skip_indices:#ignore but not remove bad rows: for example, for Alaska where there are no wells
        continue
    modified_list = [s.replace('x', '.') for s in alldf.loc[idx, 'resolution']]
    modified_list = np.array([float(item) for item in modified_list])
    lowest_index = next(idx for idx, value in enumerate(modified_list) if value == min(modified_list))
    ordered_lowest = np.argsort(modified_list)
    alldf.at[idx, 'selectedIdFromList'] = lowest_index


alldf['stateId'] = alldf['state'].map(state_abbreviations)
alldf['listIds'] = alldf['entityId']
alldf['selectedIdFromList'] = pd.to_numeric(alldf['selectedIdFromList'], errors='coerce').astype('Int64')
alldf['entityId'] = alldf.apply(lambda row: row['entityId'][row['selectedIdFromList']] if isinstance(row['entityId'], list) else row['entityId'], axis=1)
alldf['entityId'] = alldf['entityId'].apply(lambda x: x[0] if isinstance(x, list) else x)
alldf['displayId'] = alldf.apply(lambda row: row['displayId'][row['selectedIdFromList']] if isinstance(row['displayId'], list) else row['displayId'], axis=1)
alldf['displayId'] = alldf['displayId'].apply(lambda x: x[0] if isinstance(x, list) else x)
alldf['displayId'] = alldf['displayId'].str.lower()
alldf.rename(columns={'lat': 'latitude', 'long': 'longitude'}, inplace=True)
alldf = alldf[['state', 'stateId', 'accuracy', 'latitude', 'longitude', 'listIds', 'resolution', 'scenesDate', 'selectedIdFromList', 'entityId', 'displayId']]
alldf.to_pickle("positive_metadata_to_process.pkl")


#get random locations metadata
def random_location():
    # Mainland USA approximate bounding box
    min_lat, max_lat = 24.396308, 49.384358
    min_lon, max_lon = -125.001650, -66.934570
    lat = random.uniform(min_lat, max_lat)
    lon = random.uniform(min_lon, max_lon)
    return lat, lon

locations = [random_location() for _ in range(100000)]#we'll get say 100k locations
df = pd.DataFrame(locations, columns=['lat', 'long'])
df['state'] = 'Unknown' 
df['entityId'] = 'Unknown' 
df['displayId'] = 'Unknown' 
df['browsePath'] = 'Unknown' 
print(df.head())

distance_meters = 50
result_df = pd.DataFrame(columns=['index', 'lat', 'long', 'll_lat', 'll_long', 'ur_lat', 'ur_long'])
for idx, row in df.iterrows():
    ll_lat, ll_long, ur_lat, ur_long = bbox(row['lat'], row['long'])
    intersects = ((result_df['ll_lat'] <= row['lat']) & (result_df['ur_lat'] >= row['lat']) &
               (result_df['ll_long'] <= row['long']) & (result_df['ur_long'] >= row['long'])).any()
    if not intersects:
        result_df = pd.concat([result_df, pd.DataFrame([{'index': idx, 'lat': row['lat'], 'long': row['long'], 'll_lat': ll_lat, 'll_long': ll_long, 'ur_lat': ur_lat, 'ur_long': ur_long}])], ignore_index=True)


m2m = M2M("your_username", "your_password")
label='m2m-api_download'
labels = [label]
datasetName = 'naip'

nmdf = result_df
nmdf['downloadsCount'] = 'Unknown'
nmdf['browsePath'] = 'Unknown'
nmdf['entityId'] = 'Unknown'
nmdf['displayId'] = 'Unknown'
nmdf['temporalCoverage'] = 'Unknown'

for idx, row in nmdf.iterrows():
    if idx%200 == 0:
        print(idx)
        m2m = M2M("your_username", "your_password")
    params = {
        "datasetName": "naip",
        "startDate": "2015-01-01",
        "endDate": "2023-08-01",
        "boundingBox": (row['ll_long'], row['ur_long'], row['ll_lat'], row['ur_lat'])
    }
    scenes = m2m.searchScenes(**params)
    if scenes['totalHits'] == 0:
        print("Error, no recent images found. Change the date.", idx, row)
        nmdf.at[idx, 'downloadsCount'] = 0
    else:    
        nmdf.at[idx, 'downloadsCount'] = 1
        nmdf.at[idx, 'browsePath'] = scenes['results'][0]['browse'][0]['browsePath']
        nmdf.at[idx, 'entityId'] = scenes['results'][0]['entityId']
        nmdf.at[idx, 'displayId'] = scenes['results'][0]['displayId']
        nmdf.at[idx, 'temporalCoverage'] = scenes['results'][0]['temporalCoverage']
nmdf.to_pickle("negative_metadata_to_process.pkl")




