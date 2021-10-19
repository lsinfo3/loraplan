import os
import pandas as pd
import platform
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import *
import math
import threading
import time
import maputils
import numpy
import sys

#Sim
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass


import os
import math
from glob import glob
import pandas as pd
import time

import numpy as np

from numpy import arccos, dot, pi, cross
from numpy.linalg import  norm

import utm
#/Sim

sensor_file = pd.DataFrame()#used to load sensors
gateway_file = pd.DataFrame()#used to load gateways
gateway_file_grid = pd.DataFrame()#reprojected using mercator projection
manually_placed_gw_grid = pd.DataFrame()# projected manually placed gateways
sensor_file_bereinigt = pd.DataFrame()#empty lines/0 count removed
gdem_file_path = ""#DEM location
gridsize = 10#Defines the grid size in 100m increments (only tested for 10)
sensorgrid = numpy.zeros((1, 1, 1))#see below
linkgrid = numpy.zeros((1, 1, 1))#see below
textobj = None
sensorsize = 0
lock = threading.Lock()
propagation_model = "fspl"#set by ui
propagation_model_is_dynamic = False#determined by the pl model
SF = 7#set by ui
fresnel_use = True#set by ui
kml_file_path = ""#set by ui
RSSI_high = -100#set by ui
RSSI_low = -120#set by ui
use_map = False#set by ui
use_grid = False#set by ui
use_sf_sweep = False#set by ui
run_sim=False#set by ui
draw_once_completed = False #Prevents user from inserting custom gateways when no scale was set
enable_drawing=True #Set to false to prevent sensors, gateways and reachability to be drawn
sensor_file_name=""#set by ui
bulk_file_path=""#set by ui
output_file=""#set by software
widgetfont='Helvetica 11'
curr_manual_gw=1#defines the first index for manually placed gateways
# Sensorgrid is 100mX100m.
# 3-x: Sensor types, 0 GW-slag, 1 Gw-seachable sensors, 2 Gw-Index
# Linkgrid: 0:Gw-Index with best connection 1: Rx Power/RSSI in dBm gw with best connection 2:number of gateways 3: Indizes of gateways 4: lowest SF possible (0=unreachable)

def set_sim_use(sim_use_tk):
    global run_sim
    if sim_use_tk.get() == 1:
        run_sim = True
    else:
        run_sim = False

def set_rssi_thresholds(rssi_text):
    global RSSI_high
    global RSSI_low
    RSSI_high=int(rssi_text.split("/")[0])
    RSSI_low=int(rssi_text.split("/")[1])

def set_fresnel_use(fresnel_use_tk):
    global fresnel_use
    if fresnel_use_tk.get() == 1:
        fresnel_use = True
    else:
        fresnel_use = False


def set_dem_map_use(dem_map_use_tk):
    global use_map
    if dem_map_use_tk.get() == 1:
        use_map = True
    else:
        use_map = False


def set_grid_lines_use(grid_lines_use_tk):
    global use_grid
    if grid_lines_use_tk.get() == 1:
        use_grid = True
    else:
        use_grid = False


def set_sf_sweep_use(sf_sweep_use_tk):
    global use_sf_sweep
    if sf_sweep_use_tk.get() == 1:
        use_sf_sweep = True
    else:
        use_sf_sweep = False


def upload_sensor_action(event=None):
    global sensor_file
    global sensor_file_name
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = filedialog.askopenfilename(initialdir=dir_path + "\data")
    sensor_file_name=filename
    print('Sensor locations:', filename)
    sensor_file = pd.read_csv(
        filename, sep=';', encoding="ISO-8859-1", skiprows=0)
    print(sensor_file)


def upload_elevation_action(event=None):
    global gdem_file_path
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = filedialog.askdirectory(initialdir=dir_path)
    print('DEM data path:', filename)
    gdem_file_path = filename

def upload_bulk_action(event=None):
    global bulk_file_path
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = filedialog.askdirectory(initialdir=dir_path)
    print('Bulk import data path:', filename)
    bulk_file_path = filename

def upload_kml_action(event=None):
    global kml_file_path
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = filedialog.askdirectory(initialdir=dir_path)
    print('KML data path:', filename)
    kml_file_path = filename
    maputils.set_kml_file_path(kml_file_path)


def upload_gateway_action(event=None):
    global gateway_file
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = filedialog.askopenfilename(initialdir=dir_path + "\data")
    print('Gateway locations:', filename)
    gateway_file = pd.read_csv(
        filename, sep=';', encoding="ISO-8859-1", skiprows=0)
    print(gateway_file)


def setgridsize(val):
    global gridsize
    gridsize = int(val)


def setpropagation_model(value, event):
    global propagation_model
    global propagation_model_is_dynamic
    text = value.get()
    if(text == "Free space path loss"):
        propagation_model = "fspl"
        propagation_model_is_dynamic = False
    elif(text == "Manhattan"):
        propagation_model = "manhattan"
        propagation_model_is_dynamic = False
    elif(text == "Ericsson"):
        propagation_model = "ericsson"
        propagation_model_is_dynamic = True
    elif(text == "Hata"):
        propagation_model = "hata"
        propagation_model_is_dynamic = True
    elif(text == "Cost 231"):
        propagation_model = "cost231"
        propagation_model_is_dynamic = True
    elif(text == "Lee"):
        propagation_model = "lee"
        propagation_model_is_dynamic = True


def setsf(val):
    global SF
    SF = int(val)

#Ensures software uses a separate thread for ui to prevent freezes
def placement_thread(mycanvas, status,RSSI_Threshold, event=None):
    placementthr = threading.Thread(
        target=placement_action, args=(mycanvas, status,RSSI_Threshold, None,))
    placementthr.start()

#clears file
def clear_file(process_file, sensorcols):
    if(process_file.empty):
        temp = []
        indexused = False
        for index, row in process_file.iterrows():
            indexused = False
            for x in sensorcols:
                if(row[x] != 0 and indexused == False):
                    temp.append(row)
                    indexused = True
        return pd.DataFrame(temp)
    return process_file

#Combines sensors and gateways into one frame for further processing
def generate_mapframes(sensors, gws, loncols, latcols, loncolgw, latcolgw):
    global gridsize
    sensorlength = 0
    gwlength = 0
    temp = []
    for index, row in sensors.iterrows():
        newcoordpair = [row[loncols], row[latcols]]
        temp.append(newcoordpair)
        sensorlength += 1
    for index, row in gateway_file.iterrows():
        newcoordpair = [row[loncolgw], row[latcolgw]]
        temp.append(newcoordpair)
        gwlength += 1
    togenerate = pd.DataFrame(temp)
    mapframe_all = maputils.generatescreenmap(
        togenerate, 0, 1, (gridsize/10000))
    temps = []
    tempgw = []
    for index, row in mapframe_all.iterrows():
        if (index < sensorlength):
            newcoordpair = [row[0], row[1]]
            temps.append(newcoordpair)
        else:
            newcoordpair = [row[0], row[1]]
            tempgw.append(newcoordpair)
    #contains all x,y Coordinates from Sensors (not combined to a single grid)
    mapframe = pd.DataFrame(temps)
    mapframe_gw = pd.DataFrame(tempgw)
    return [mapframe, mapframe_gw, sensorlength, gwlength]

#Clears sensor and linkgrid and expands them
def setup_sensorgrid(mapframes, sensorsize):
    global sensorgrid
    global gridsize
    global linkgrid
    maxx = 0
    maxy = 0
    for i in range(0, 1):
        for index, row in mapframes[i].iterrows():
            if(row[0] > maxx):
                maxx = int(row[0]+1)
            if(row[1] > maxy):
                maxy = int(row[1]+1)
    sizex = int((maxx*((1/(gridsize))*100))+1)
    sizey = int((maxy*((1/(gridsize))*100))+1)
    sensorgrid = numpy.zeros((sizex, sizey, sensorsize+3))
    linkgrid = [[[0 for z in range(0, 5)] for y in range(0, sizey)]
                for x in range(0, sizex)]
    for x in range(0, sizex):
        for y in range(0, sizey):
            linkgrid[x][y][0] = 0
            linkgrid[x][y][1] = -math.inf
            linkgrid[x][y][2] = 0
            linkgrid[x][y][3] = []
            linkgrid[x][y][4] = 100

#sets the range for FSPL and manhattan models
def setpersistantrange():
    global SF
    global propagation_model
    if(propagation_model == "fspl"):
        fsplrange = [970, 1370, 1940, 2740, 3080, 4350]
        return fsplrange[SF-7]
    elif(propagation_model == "manhattan"):
        return 10
    else:
        return 10

#generates the csv export and the input for the simulation
def csv_export():
    global sensorgrid
    global linkgrid
    global propagation_model
    global sensorsize
    global sensor_file_name
    global output_file
    output = {"lon": [], "lat": [], "BestGW": [], "RSSI": [],
              "SF": [], "NumberOfSensors": [], "OtherGWs": []}
    sizex = sensorgrid.shape[0]
    sizey = sensorgrid.shape[1]
    for x in range(0, sizex):
        for y in range(0, sizey):
            if(linkgrid[x][y][2] != 0):
                scalefactor = (1/100)
                mx = (x/scalefactor)+maputils.minx
                my = (y/scalefactor)+maputils.miny
                output['lat'].append(maputils.gudermann(
                    mx, my, maputils.centerlon)[1])
                output['lon'].append(maputils.gudermann(
                    mx, my, maputils.centerlon)[0])
                output['BestGW'].append(
                    str(gateway_file_grid.iloc[int(linkgrid[x][y][0])][0]))
                output['RSSI'].append(linkgrid[x][y][1])
                output['SF'].append(linkgrid[x][y][4])
                numsensors = 0
                for i in range(0, sensorsize):
                    numsensors += sensorgrid[x][y][i+3]
                output['NumberOfSensors'].append(numsensors)
                othergws = [str(gateway_file_grid.iloc[int(curr)][0])
                            for curr in linkgrid[x][y][3]]
                output['OtherGWs'].append(othergws)
    pd.DataFrame(data=output).to_csv(sensor_file_name.split("/")[-1].split(".")[0]+
        '_reachable_sensors_'+propagation_model+'.csv')
    pd.DataFrame(data=output).to_json(sensor_file_name.split("/")[-1].split(".")[0]+
        '_reachable_sensors_'+propagation_model+'.json')
    output_file=output

#set the contrast for drawing a DEM map
def dembackground(mycanvas, sensorgrid):
    global gridsize
    global gdem_file_path
    if(gdem_file_path != ""):
        sizex = numpy.size(sensorgrid, 0)
        sizey = numpy.size(sensorgrid, 1)
        maxintensity = 0
        minintensity = sys.maxsize
        for y in range(0, sizey):
            for x in range(0, sizex):
                curr = maputils.getelevation_from_map(gdem_file_path, x, y)
                if(curr > maxintensity):
                    maxintensity = curr
                if(curr < minintensity):
                    minintensity = curr
        return([maxintensity, minintensity])

# distance helper function for simulation
def distance_numpy(A, B, P):
    """
    Created on Tue Mar 30 14:45:57 2021

    @author: frank
    """
    """ segment line AB, point P, where each one is an array([x, y]) """
    if all(A == P) or all(B == P):
        return 0
    if arccos(dot((P - A) / norm(P - A), (B - A) / norm(B - A))) > pi / 2:
        return norm(P - A)
    if arccos(dot((P - B) / norm(P - B), (A - B) / norm(A - B))) > pi / 2:
        return norm(P - B)
    return norm(cross(A-B, A-P))/norm(B-A)
#Simulate the collision probability and generate export
def sim_collision_probability(status):
    global output_file
    global gateway_file_grid
    global sensorgrid
    global propagation_model
    global sensor_file_name
    """
    Created on Tue Mar 30 14:45:57 2021

    @author: frank
    """
    pd.options.display.float_format = '{:.2f}'.format

    gateway_height = 5.0
    gateway_env = 'urban'

    sf = [7,8,9,10,11,12]
    hata_distances = np.zeros(len(sf))

    #get transmission distances by model
    for i in range(len(sf)):
        if(propagation_model == "fspl"):
            hata_distances[i] = [970, 1370, 1940, 2740, 3080, 4350][i]
        elif(propagation_model == "manhattan"):
            hata_distances[i] = 10
        elif(propagation_model == "ericsson"):
            hata_distances[i] = maputils.ericsson(sensorgrid,0,0,0,gateway_file, sf[i])
        elif(propagation_model == "hata"):
            hata_distances[i] = maputils.hata(sensorgrid,0,0,0,gateway_file, sf[i])
        elif(propagation_model == "cost231"):
            hata_distances[i] = maputils.cost231(sensorgrid,0,0,0,gateway_file, sf[i])
        elif(propagation_model == "lee"):
            hata_distances[i] = maputils.lee(sensorgrid,0,0,0,gateway_file, sf[i])
    #%% 
    sensor_data = output_file
    gateway_data={}
    sensor_data['distance'] = np.zeros(len(output_file['lon']))      
    sensor_data['range'] = np.zeros(len(output_file['lon']))      
    sensor_data['sf_collisions'] = np.empty((len(output_file['lon']), 0)).tolist()
    sensor_data['gw_x'] = np.zeros(len(output_file['lon'])) 
    sensor_data['gw_y'] = np.zeros(len(output_file['lon'])) 
    sensor_data['sen_x'] = np.zeros(len(output_file['lon'])) 
    sensor_data['sen_y'] = np.zeros(len(output_file['lon'])) 

    for j in range(len(output_file['lon'])):
        bestgw=str(sensor_data["BestGW"][j])
        temp_gw_file=[]
        for i in range(len(gateway_file_grid)):
            temp_gw_file.append(str(gateway_file_grid.iloc[i][0]))
        scalefactor = (1/100)
        mx = (gateway_file_grid.iloc[temp_gw_file.index(bestgw)][1]/scalefactor)+maputils.minx
        my = (gateway_file_grid.iloc[temp_gw_file.index(bestgw)][2]/scalefactor)+maputils.miny
        latlon=maputils.gudermann(mx,my,maputils.centerlon)
        tmp = utm.from_latlon(latlon[0],latlon[1])
        sensor_data['gw_y'][j] = tmp[0]
        sensor_data['gw_x'][j] = tmp[1]

    for j in range(len(output_file['lon'])):
        sensor_data['sf_collisions'][j] = [0,0,0,0,0,0]
        sensor_data['distance'][j] = hata_distances[sensor_data['SF'][j]-7]
                            
        tmp = utm.from_latlon(sensor_data['lat'][j], sensor_data['lon'][j])
        sensor_data["sen_y"][j] = tmp[0]
        sensor_data["sen_x"][j] = tmp[1]
        
    for j in range(len(output_file['lon'])):
        status.config(state=NORMAL)
        status.delete('1.0', END)
        status.insert(
            END, "Visualization in Progress...\nPreparing simulation ("+str(int((j/len(sensor_data['lon']))*100))+"%)")
        status.config(state=DISABLED)
        if(sensor_data['NumberOfSensors'][j] > 1):
            sensor_data['range'][j] += sensor_data['NumberOfSensors'][j]-1
            sensor_data['sf_collisions'][j][sensor_data['SF'][j]-7] += sensor_data['NumberOfSensors'][j]-1
        
        for l in range(j+1, len(output_file['lon'])):
            if(l != j):
                p1 = (sensor_data['sen_x'][j], sensor_data['sen_y'][j])
                p1 = np.asarray(p1)
                p2 = (sensor_data['gw_x'][j], sensor_data['gw_y'][j])
                p2 = np.asarray(p2)
                p3 = (sensor_data['sen_x'][l], sensor_data['sen_y'][l])
                p3 = np.asarray(p3)
                
                dist = distance_numpy(p1,p2,p3)
                
                sensor_dist = math.hypot(sensor_data['sen_x'][j] - sensor_data['sen_x'][l], sensor_data['sen_y'][j] - sensor_data['sen_y'][l])
                
                if(sensor_dist < sensor_data['distance'][l]):
                    sensor_data['range'][j] +=  sensor_data['NumberOfSensors'][l]
                    sensor_data['sf_collisions'][j][sensor_data['SF'][l]-7] += sensor_data['NumberOfSensors'][l]
                elif(dist < sensor_data['distance'][l]):
                    sensor_data['range'][j] +=  sensor_data['NumberOfSensors'][l]
                    sensor_data['sf_collisions'][j][sensor_data['SF'][l]-7] += sensor_data['NumberOfSensors'][l]
    #lora related parameters
    payload_bytes = 8
    cyclic_redundancy_check = 1
    header_enabled = 1
    header_length = 20
    low_datarate_optimize = 0
    coding_rate = 4
    preamble_length = 8
    sim_reruns = 1
    sim_accuracy = 4 #number of after comma positions
    simtime = 3600 #3600 equals to 1 msg per hour

    messages_per_hour = 1
    bandwidth = 125000
    sf = [7, 8, 9, 10, 11, 12]
    toa = np.zeros(len(sf))
    for i in range(len(sf)):
        all_packet = (8 * payload_bytes - (4*sf[i]) + 8 + 16 * cyclic_redundancy_check + 20 * header_enabled) / (4 * sf[i] - 2*low_datarate_optimize)
        n_packet = 8 + np.max((np.ceil(all_packet)* (coding_rate + 4)), 0)
        total_symbols = preamble_length + 4.25 + n_packet
        symbol_duration = (2**sf[i])/bandwidth
        toa[i] = symbol_duration * total_symbols


    collision_dataframe = pd.DataFrame()

    #%%
    collision_probab_list = np.zeros(1)
    all_collision_probabs = [[]] * 1
    total_sensors_in_range = np.sum(sensor_data['NumberOfSensors'])
    tmp_collision_probab_list = np.zeros(sim_reruns)
    for k in range(sim_reruns):
        current_sensor_counter = 0
        for j in range(len(sensor_data['lon'])):
            status.config(state=NORMAL)
            status.delete('1.0', END)
            status.insert(
                END, "Visualization in Progress...\nSimulating collision probability ("+str(int((j/len(sensor_data['lon']))*100))+"%)")
            status.config(state=DISABLED)
            transmission_starts_this_entry = np.round(np.random.uniform(low = 0, high=3600, size=int(sensor_data['NumberOfSensors'][j])), sim_accuracy)
            transmission_starts_other_sensors = np.round(np.random.uniform(low = 0, high=3600, size=int(sensor_data['range'][j])), sim_accuracy)
            sf_collisions = sensor_data['sf_collisions'][j]
            for l in range(len(transmission_starts_this_entry)):
                current_sensor_start = transmission_starts_this_entry[l]
                current_sensor_end = current_sensor_start + toa[sensor_data['SF'][j] - 7]
                toas_sensors_sf7 = np.zeros(int(sf_collisions[0])) + toa[0]
                toas_sensors_sf8 = np.zeros(int(sf_collisions[1])) + toa[1]
                toas_sensors_sf9 = np.zeros(int(sf_collisions[2])) + toa[2]
                toas_sensors_sf10 = np.zeros(int(sf_collisions[3])) + toa[3]
                toas_sensors_sf11 = np.zeros(int(sf_collisions[4])) + toa[4]
                toas_sensors_sf12 = np.zeros(int(sf_collisions[5])) + toa[5]
                
                toas_other_sensors = np.concatenate((toas_sensors_sf7, toas_sensors_sf8, toas_sensors_sf9, toas_sensors_sf10, toas_sensors_sf11, toas_sensors_sf12), axis=None)
                
                transmission_end_other_sensors = transmission_starts_other_sensors + toas_other_sensors
                
                other_sensors_df = pd.DataFrame({"start": transmission_starts_other_sensors, "end": transmission_end_other_sensors})
                other_sensors_df = other_sensors_df.sort_values('start')
                if len(np.where((other_sensors_df['start'] < current_sensor_start) & (other_sensors_df['end'] > current_sensor_start))[0]) > 0:
                    current_sensor_counter +=1
                elif len(np.where((other_sensors_df['start'] < current_sensor_end) & (other_sensors_df['end'] > current_sensor_end))[0]) > 0:
                    current_sensor_counter += 1
        
        tmp_collision_probab_list[k] = current_sensor_counter/total_sensors_in_range
    return(tmp_collision_probab_list)

#Main function used for the calculation
def start_calculation(mycanvas,status):
    global sensor_file_bereinigt
    global gateway_file
    global gateway_file_grid
    global manually_placed_gw_grid
    global sensorgrid
    global linkgrid
    global gridsize
    global sensorsize
    global SF
    global propagation_model
    global propagation_model_is_dynamic
    global gdem_file_path
    global fresnel_use
    global RSSI_high
    global RSSI_low
    global use_map
    global use_grid
    global use_sf_sweep
    global draw_once_completed
    global run_sim
    print('starting calculation')
    lock.acquire()
    mycanvas.delete("all")
    progresstext = mycanvas.create_text((mycanvas.winfo_width()/2), (mycanvas.winfo_height(
    )/2), text="Visualization in Progress...\nDo not change any values!", font='Helvetica 14 bold')
    status.config(state=NORMAL)
    status.delete('1.0', END)
    status.insert(
        END, "Visualization in Progress...\nDo not change any values!")
    status.config(state=DISABLED)
    sizex = int(mycanvas.winfo_width())
    sizey = int(mycanvas.winfo_height())
    total_reachable_sensors = 0
    total_sensors = 0
    sensorcolumnsinfile = [2]
    sensorsize = len(sensorcolumnsinfile)
    sensor_file_bereinigt = clear_file(sensor_file, sensorcolumnsinfile)
    print("file cleared of empty sensors")
    print(sensor_file_bereinigt)
    mapframes = generate_mapframes(
        sensor_file_bereinigt, gateway_file, 1, 0, 2, 1)
    mapframe = mapframes[0]
    mapframe_gw = mapframes[1]
    setup_sensorgrid(mapframes, sensorsize)
    print("drawing sensors...")
    status.config(state=NORMAL)
    status.delete('1.0', END)
    status.insert(
        END, "Visualization in Progress...\nDrawing sensors...")
    status.config(state=DISABLED)
    workfile = sensor_file_bereinigt.values.tolist()
    workfile_gw = gateway_file.values.tolist()
    temp = []
    x = 0
    y = 0
    for index, row in mapframe_gw.iterrows():  # assigns GWs a grid cell
        x = int((row[0]//(gridsize/100)))
        y = int((row[1]//(gridsize/100)))
        currgw = [workfile_gw[index][0], x, y]
        temp.append(currgw)
    temp_df = pd.DataFrame(temp)
    gateway_file_grid = temp_df
    gateway_file_grid = gateway_file_grid.append(
        manually_placed_gw_grid, ignore_index=True)
    maxx = 0
    maxy = 0
    offsetx = int(math.ceil((sizex/2) / (gridsize/100))) * (gridsize/100)
    offsety = int(math.ceil((sizey/2) / (gridsize/100))) * (gridsize/100)
    mycanvas.delete("all")
    for index, row in mapframe.iterrows():
        x = int((row[0]//(gridsize/100)))
        y = int((row[1]//(gridsize/100)))
        for i in range(0, sensorsize):
            if i < len(sensorcolumnsinfile):
                sensorgrid[x][y][i+3] += int(workfile[index]
                                             [sensorcolumnsinfile[i]])
        if(x > maxx):
            maxx = x
        if(y > maxy):
            maxy = y
        todraw = 0
        for i in range(0, sensorsize):
            todraw += sensorgrid[0][0][i+3]
        if(todraw > 0):
            draw_rect_in_grid(mycanvas, x, y, gridsize,
                              offsetx, offsety, 'deep sky blue')
        # this is neccesary for very small datasets that only fill one cell
    mycanvas.config(width=(maxx*gridsize/10), height=maxy*(gridsize/10),
                    scrollregion=(0, 0, maxx*(gridsize/10), maxy*(gridsize/10)))
    if gdem_file_path != "" and use_map:
        demmaxmin = dembackground(mycanvas, sensorgrid)
    for x in range(maxx+1):
        for y in range(maxy+1):
            for i in range(0, sensorsize):
                total_sensors += sensorgrid[x][y][i+3]
            if gdem_file_path != "" and use_map:
                curr = maputils.getelevation_from_map(gdem_file_path, x, y)
                curr = int(
                    ((curr-demmaxmin[1])/(demmaxmin[0]-demmaxmin[1]))*100)
                color = 'grey'+str(curr)
                draw_rect_in_grid(mycanvas, x, y, gridsize, 0, 0, color)
            todraw = 0
            for i in range(0, sensorsize):
                todraw += sensorgrid[x][y][i+3]
            if(todraw > 0):
                draw_rect_in_grid(mycanvas, x, y, gridsize,
                                  offsetx, offsety, 'gray80')
    print("drawing gateways...")
    status.config(state=NORMAL)
    status.delete('1.0', END)
    status.insert(
        END, "Visualization in Progress...\nDrawing gateways...")
    status.config(state=DISABLED)
    usedsensors = []
    maputils.set_fresnel_use(fresnel_use)
    for index, row in gateway_file_grid.iterrows():
        status.config(state=NORMAL)
        status.delete('1.0', END)
        status.insert(
            END, "Visualization in Progress...\nDrawing gateways ("+str(index+1)+"/"+str(len(gateway_file_grid))+")")
        status.config(state=DISABLED)
        sensorgrid[int(row[1])][int(row[2])][0] = 1
        sensorgrid[int(row[1])][int(row[2])][2] = index
        if(propagation_model_is_dynamic):
            if(use_sf_sweep):
                for SF_curr in range(7, 13):
                    reachablesensors = maputils.calculatereachability_with_model(sensorgrid, linkgrid, gdem_file_path, int(
                        row[1]), int(row[2]), sensorsize, propagation_model, gateway_file, SF_curr)
            else:
                reachablesensors = maputils.calculatereachability_with_model(sensorgrid, linkgrid, gdem_file_path, int(
                    row[1]), int(row[2]), sensorsize, propagation_model, gateway_file, SF)
        else:
            if(use_sf_sweep):
                for SF_curr in range(7, 13):
                    reachablesensors = maputils.calculatereachability(sensorgrid, linkgrid, gdem_file_path, int(
                        row[1]), int(row[2]), sensorsize, setpersistantrange(), propagation_model, gateway_file, SF_curr)
            else:
                reachablesensors = maputils.calculatereachability(
                    sensorgrid, linkgrid, gdem_file_path, int(row[1]), int(row[2]), sensorsize, setpersistantrange(), propagation_model, gateway_file, SF)
        for sensorcoord in range(len(reachablesensors)):
            xs = reachablesensors[sensorcoord][0]
            ys = reachablesensors[sensorcoord][1]
            usedsensors.append([xs, ys])
            if(sensorgrid[xs][ys][0] == 1):
                draw_rect_in_grid(mycanvas, xs, ys, gridsize,
                                  offsetx, offsety, 'indigo')
            else:
                if(linkgrid[xs][ys][2] > 1):
                    draw_rect_in_grid(mycanvas, xs, ys,
                                      gridsize, offsetx, offsety, 'cyan')
                else:
                    if(linkgrid[xs][ys][1] >= RSSI_high):
                        draw_rect_in_grid(
                            mycanvas, xs, ys, gridsize, offsetx, offsety, 'green')
                    elif(linkgrid[xs][ys][1] >= RSSI_low):
                        draw_rect_in_grid(mycanvas, xs, ys,
                                          gridsize, offsetx, offsety, 'orange')
                    else:
                        draw_rect_in_grid(mycanvas, xs, ys,
                                          gridsize, offsetx, offsety, 'red')

    uniquereachablesensors = []
    for x in usedsensors:
        if x not in uniquereachablesensors:
            uniquereachablesensors.append(x)
    for x in range(len(uniquereachablesensors)):
        for s in range(0, sensorsize):
            total_reachable_sensors += sensorgrid[uniquereachablesensors[x]
                                                  [0]][uniquereachablesensors[x][1]][s+3]
    print("exporting to csv...")
    status.config(state=NORMAL)
    status.delete('1.0', END)
    status.insert(
        END, "Visualization in Progress...\nExporting to csv...")
    status.config(state=DISABLED)
    csv_export()
    if(run_sim):
        print("simulating collision probability...")
        status.config(state=NORMAL)
        status.delete('1.0', END)
        status.insert(
            END, "Visualization in Progress...\nPreparing simulation...")
        status.config(state=DISABLED)
        sim_results=sim_collision_probability(status)[0]
    if(gridsize >= 50 and not use_map and use_grid):
        print("drawing grid...")
        status.config(state=NORMAL)
        status.delete('1.0', END)
        status.insert(
            END, "Visualization in Progress...\nDrawing grid...")
        status.config(state=DISABLED)
        currx = 0
        curry = 0
        while (currx < maxx*(gridsize)):
            mycanvas.update_idletasks()
            mycanvas.create_line(currx, 0, currx, maxx *
                                 (gridsize), fill='grey85')
            currx += gridsize/10
        while (curry < maxy*(gridsize)):
            mycanvas.update_idletasks()
            mycanvas.create_line(0, curry, maxy*(gridsize),
                                 curry, fill='grey85')
            curry += gridsize/10
    status.config(state=NORMAL)
    status.delete('1.0', END)
    if(run_sim):
        status.insert(END, "total number of sensors: " + str(int(total_sensors)) + "\n" +
                    "reachable sensors from all gateways: " + str(int(total_reachable_sensors))+"\n"+
                    "collision probability in this network: "+str("%.2f" % (sim_results*100))+"%")
    else:
        status.insert(END, "total number of sensors: " + str(int(total_sensors)) + "\n" +
                    "reachable sensors from all gateways: " + str(int(total_reachable_sensors)))
    status.config(state=DISABLED)
    draw_once_completed = True
    print("done.")
    lock.release()

#function called by the visualize button. Sets up appropriate options
def placement_action(mycanvas, status,RSSI_Selector, event=None):
    global bulk_file_path
    global sensor_file
    global sensor_file_name
    global gateway_file
    global gateway_file_grid
    global manually_placed_gw_grid
    global sensor_file_bereinigt
    global sensorgrid
    global linkgrid
    global textobj
    global sensorsize
    set_rssi_thresholds(RSSI_Selector.get(1.0, 'end-1c'))
    if(bulk_file_path!=""):
        for file_name in os.listdir(bulk_file_path):
            if file_name.endswith("bulksensors.csv"):
                sensor_file = pd.read_csv(bulk_file_path+"/"+file_name, sep=';', encoding="ISO-8859-1", skiprows=0)
                sensor_file_name=bulk_file_path+"/"+file_name
                x=file_name.split("bulk")
                currentset=x[0]
                print("Bulk file import currently processing: "+str(currentset))
                gw_file_path= str(bulk_file_path)+"/"+str(currentset)+"bulkgateways.csv"
                gateway_file = pd.read_csv(gw_file_path, sep=';', encoding="ISO-8859-1", skiprows=0)
                start_calculation(mycanvas,status)
                sensor_file = pd.DataFrame()
                gateway_file = pd.DataFrame()
                gateway_file_grid = pd.DataFrame()
                manually_placed_gw_grid = pd.DataFrame()
                sensor_file_bereinigt = pd.DataFrame()
                sensorgrid = numpy.zeros((1, 1, 1))
                linkgrid = numpy.zeros((1, 1, 1))
                textobj = None
                sensorsize = 0
                maputils.use_1m_dem = False
    else:
        start_calculation(mycanvas,status)

#Action for the left click in the drawn map
def left_click(mycanvas, radiobutton, event):
    global gateway_file_grid
    global gridsize
    global textobj
    global sensorgrid
    global manually_placed_gw_grid
    global sensorsize
    global draw_once_completed
    global curr_manual_gw
    if(radiobutton.get() == 1 and draw_once_completed):
        print(event)
        mycanvas.create_oval(mycanvas.canvasx(event.x), mycanvas.canvasy(
            event.y), (mycanvas.canvasx(event.x)+10), (mycanvas.canvasy(event.y)+10), fill='red')
        new_gw = pd.DataFrame([[str('manually placed'+str(curr_manual_gw)), int((mycanvas.canvasx(
            event.x)//(gridsize/10))), int((mycanvas.canvasy(event.y)//(gridsize/10)))]])
        manually_placed_gw_grid = manually_placed_gw_grid.append(new_gw)
        curr_manual_gw+=1
    else:
        print(event)
        mycanvas.delete(textobj)
        sgx = int((mycanvas.canvasx(event.x)//(gridsize/10)))
        sgy = int((mycanvas.canvasy(event.y)//(gridsize/10)))
        if(sensorgrid[sgx][sgy][0] == 1):
            textobj = mycanvas.create_text(mycanvas.canvasx(event.x), mycanvas.canvasy(
                event.y), anchor='s', text="sensors reachable from gateway \"" + str(gateway_file_grid.iloc[int(sensorgrid[sgx][sgy][2])][0]) + "\": " + str(int(sensorgrid[sgx][sgy][1])), font='Helvetica 11 bold')
        else:
            numsensors = 0
            for i in range(0, sensorsize):
                numsensors += sensorgrid[sgx][sgy][i+3]
            try:
                textobj = mycanvas.create_text(mycanvas.canvasx(event.x), mycanvas.canvasy(
                    event.y), anchor='s', text="number of sensors: "+str(int(numsensors)) + "\n reachable via "+str(int(linkgrid[sgx][sgy][2]))+" Gateway(s). Best RSSI: \n\"" + str(gateway_file_grid.iloc[int(linkgrid[sgx][sgy][0])][0])+"\" with RSSI: " + str(int(linkgrid[sgx][sgy][1]))+"dBm with SF: " + str(int(linkgrid[sgx][sgy][4])), font='Helvetica 11 bold')
            except:
                textobj = mycanvas.create_text(mycanvas.canvasx(event.x), mycanvas.canvasy(
                    event.y), anchor='s', text="number of sensors: "+str(int(numsensors)), font='Helvetica 11 bold')

#helper function for drawing
def draw_rect_in_grid(mycanvas, x, y, gridsize, offsetx, offsety, color):
    global enable_drawing
    if enable_drawing:
        mycanvas.update_idletasks()
        mycanvas.create_rectangle(x*(gridsize/10), y*(gridsize/10), (x*(gridsize/10)+(
            gridsize/10)), (y*(gridsize/10)+(gridsize/10)), fill=color, outline=color)

#reset all critical parameters (executed by the reset all button)
def reset_all():
    global sensor_file
    global gateway_file
    global gateway_file_grid
    global manually_placed_gw_grid
    global sensor_file_bereinigt
    global gdem_file_path
    global sensorgrid
    global linkgrid
    global textobj
    global sensorsize
    global kml_file_path
    global sensor_file
    global bulk_file_path
    global curr_manual_gw

    sensor_file = pd.DataFrame()
    gateway_file = pd.DataFrame()
    gateway_file_grid = pd.DataFrame()
    manually_placed_gw_grid = pd.DataFrame()
    sensor_file_bereinigt = pd.DataFrame()
    gdem_file_path = ""
    sensorgrid = numpy.zeros((1, 1, 1))
    linkgrid = numpy.zeros((1, 1, 1))
    textobj = None
    sensorsize = 0
    kml_file_path = ""
    maputils.use_1m_dem = False
    bulk_file_path=""
    curr_manual_gw=1

#generates the ui
def create_widgets(self, main_container):
    global widgetfont
    right_frame = tk.Frame(main_container,
                           width=350,
                           height=main_container.winfo_screenheight(),
                           highlightbackground="black",
                           highlightthickness=1)

    left_frame = tk.Frame(main_container,

                          highlightbackground="black",
                          highlightthickness=1)

    left_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
    right_frame.pack(side=tk.RIGHT, ipadx=5, expand=False, fill=tk.BOTH)
    mycanvas = RSCanvas(left_frame, width=main_container.winfo_screenwidth(
    ), height=main_container.winfo_screenheight(), bg="white", highlightthickness=0, scrollregion=(0, 0, 0, 0))
    untereleiste = Scrollbar(left_frame, orient=HORIZONTAL)
    rechteleiste = Scrollbar(left_frame, orient=VERTICAL)
    untereleiste.config(command=mycanvas.xview)
    rechteleiste.config(command=mycanvas.yview)
    untereleiste.pack(side=BOTTOM, fill=X)
    rechteleiste.pack(side=RIGHT, fill=Y)
    mycanvas.config(xscrollcommand=untereleiste.set,
                    yscrollcommand=rechteleiste.set)
    mycanvas.pack(fill=BOTH, expand=YES)
    # http://effbot.org/zone/tkinter-scrollbar-patterns.htm was helpfull
    parameter_label = tk.Label(
        right_frame, text="Parameters:", font='Helvetica 16 bold')
    parameter_label.grid(row=0, column=0, sticky='nw')

    sensor_coordinates_label = tk.Label(right_frame, text='Sensor locations', font=widgetfont)
    self.sensor_coordinates_upload = tk.Button(
        right_frame, text='Open', command=upload_sensor_action, font=widgetfont)

    elevation_label = tk.Label(right_frame, text='DEM data path (optional)', font=widgetfont)
    self.sensor_elevation_upload = tk.Button(
        right_frame, text='Open', command=upload_elevation_action, font=widgetfont)
    elevation_label.grid(row=4, column='0', sticky='nw', pady=5)

    kml_label = tk.Label(
        right_frame, text='KML data path (optional, requires DEM)', font=widgetfont)
    self.kml_upload = tk.Button(
        right_frame, text='Open', command=upload_kml_action, font=widgetfont)
    kml_label.grid(row=5, column='0', sticky='nw', pady=5)

    sensor_coordinates_label.grid(row=1, column='0', sticky='nw', pady=5)
    self.sensor_coordinates_upload.grid(row=1, column='1', sticky='ne', pady=5)
    self.sensor_elevation_upload.grid(row=4, column='1', sticky='ne', pady=5)
    self.kml_upload.grid(row=5, column='1', sticky='ne', pady=5)

    self.manual_sensor_placement = tk.IntVar()
    self.manual_gateway_placement = tk.IntVar()
    mycanvas.bind("<Button-1>", lambda event, mc=mycanvas,
                  gw=self.manual_gateway_placement: left_click(mc, gw, event))
   
    gateway_coordinates_label = tk.Label(right_frame, text='Gateway locations', font=widgetfont)
    gateway_coordinates_upload = tk.Button(
        right_frame, text='Open', command=upload_gateway_action, font=widgetfont)

    gateway_coordinates_label.grid(row=2, column=0, sticky='nw', pady=5)
    gateway_coordinates_upload.grid(row=2, column=1, sticky='ne', pady=5)

    bulk_upload_label = tk.Label(right_frame, text='Bulk processing (overrides sensor/gateway selection)', font=widgetfont)
    bulk_upload = tk.Button(
        right_frame, text='Open', command=upload_bulk_action, font=widgetfont)

    bulk_upload_label.grid(row=3, column=0, sticky='nw', pady=5)
    bulk_upload.grid(row=3, column=1, sticky='ne', pady=5)

    gateway_coordinates_distribution_label = tk.Label(
        right_frame, text='Manual GW placement', font=widgetfont)
    gateway_coordinates_distribution_frame = tk.Frame(right_frame)

    gateway_coordinates_distribution_label.grid(
        row=7, column=0, sticky='nw', pady=5)
    gateway_coordinates_distribution_frame.grid(
        row=7, column=1, sticky='ne', pady=5)

    gateway_coordinates_button1 = tk.Radiobutton(gateway_coordinates_distribution_frame,
                                                 text="yes",
                                                 variable=self.manual_gateway_placement,
                                                 value=1, font=widgetfont)
    gateway_coordinates_button2 = tk.Radiobutton(gateway_coordinates_distribution_frame,
                                                 text="no",
                                                 variable=self.manual_gateway_placement,
                                                 value=2, font=widgetfont)
    gateway_coordinates_button2.select()

    gateway_coordinates_button1.grid(row=0, column=0)
    gateway_coordinates_button2.grid(row=0, column=1)

    RSSI_label = tk.Label(
        right_frame, text='Good/Bad RSSI threshold (dBm)', font=widgetfont)
    RSSI_label.grid(
        row=6, column=0, sticky='nw', pady=5)
    RSSI_Threshold= Text(right_frame, height=1, width=1, font=widgetfont)
    RSSI_Threshold.insert(1.0,"-100/-120")
    RSSI_Threshold.grid(row=6, column=1, columnspan=1, sticky='nesw', pady=10)

    sim_label = tk.Label(
        right_frame, text='Simulate collision probability', font=widgetfont)
    sim_frame = tk.Frame(right_frame)
    sim_label.grid(
        row=8, column=0, sticky='nw', pady=5)
    sim_frame.grid(
        row=8, column=1, sticky='ne', pady=5)
    self.sim_use_tk = tk.IntVar()
    sim_button1 = tk.Radiobutton(sim_frame,
                                     text="yes",
                                     variable=self.sim_use_tk,
                                     command=lambda fr=self.sim_use_tk: set_sim_use(
                                         fr),
                                     value=1, font=widgetfont)
    sim_button2 = tk.Radiobutton(sim_frame,
                                     text="no",
                                     variable=self.sim_use_tk,
                                     command=lambda fr=self.sim_use_tk: set_sim_use(
                                         fr),
                                     value=2, font=widgetfont)
    sim_button2.select()

    sim_button1.grid(row=0, column=0)
    sim_button2.grid(row=0, column=1)

    fresnel_label = tk.Label(
        right_frame, text='Use 2D Fresnel Zone', font=widgetfont)
    fresnel_frame = tk.Frame(right_frame)
    fresnel_label.grid(
        row=9, column=0, sticky='nw', pady=5)
    fresnel_frame.grid(
        row=9, column=1, sticky='ne', pady=5)
    self.fresnel_use_tk = tk.IntVar()
    fresnel_button1 = tk.Radiobutton(fresnel_frame,
                                     text="yes",
                                     variable=self.fresnel_use_tk,
                                     command=lambda fr=self.fresnel_use_tk: set_fresnel_use(
                                         fr),
                                     value=1, font=widgetfont)
    fresnel_button2 = tk.Radiobutton(fresnel_frame,
                                     text="no",
                                     variable=self.fresnel_use_tk,
                                     command=lambda fr=self.fresnel_use_tk: set_fresnel_use(
                                         fr),
                                     value=2, font=widgetfont)
    fresnel_button2.select()

    fresnel_button1.grid(row=0, column=0)
    fresnel_button2.grid(row=0, column=1)

    dem_map_label = tk.Label(
        right_frame, text='Draw DEM map as background (slow)', font=widgetfont)
    dem_map_frame = tk.Frame(right_frame)
    dem_map_label.grid(
        row=10, column=0, sticky='nw', pady=5)
    dem_map_frame.grid(
        row=10, column=1, sticky='ne', pady=5)
    self.dem_map_use_tk = tk.IntVar()
    dem_map_button1 = tk.Radiobutton(dem_map_frame,
                                     text="yes",
                                     variable=self.dem_map_use_tk,
                                     command=lambda fr=self.dem_map_use_tk: set_dem_map_use(
                                         fr),
                                     value=1, font=widgetfont)
    dem_map_button2 = tk.Radiobutton(dem_map_frame,
                                     text="no",
                                     variable=self.dem_map_use_tk,
                                     command=lambda fr=self.dem_map_use_tk: set_dem_map_use(
                                         fr),
                                     value=2, font=widgetfont)
    dem_map_button2.select()

    dem_map_button1.grid(row=0, column=0)
    dem_map_button2.grid(row=0, column=1)

    grid_lines_label = tk.Label(
        right_frame, text='Draw Grid lines', font=widgetfont)
    grid_lines_frame = tk.Frame(right_frame)
    grid_lines_label.grid(
        row=11, column=0, sticky='nw', pady=5)
    grid_lines_frame.grid(
        row=11, column=1, sticky='ne', pady=5)
    self.grid_lines_use_tk = tk.IntVar()
    grid_lines_button1 = tk.Radiobutton(grid_lines_frame,
                                        text="yes",
                                        variable=self.grid_lines_use_tk,
                                        command=lambda fr=self.grid_lines_use_tk: set_grid_lines_use(
                                            fr),
                                        value=1, font=widgetfont)
    grid_lines_button2 = tk.Radiobutton(grid_lines_frame,
                                        text="no",
                                        variable=self.grid_lines_use_tk,
                                        command=lambda fr=self.grid_lines_use_tk: set_grid_lines_use(
                                            fr),
                                        value=2, font=widgetfont)
    grid_lines_button2.select()

    grid_lines_button1.grid(row=0, column=0)
    grid_lines_button2.grid(row=0, column=1)

    sf_sweep_label = tk.Label(
        right_frame, text='Sweep over all SFs (ignores manual SF selection)', font=widgetfont)
    sf_sweep_frame = tk.Frame(right_frame)
    sf_sweep_label.grid(
        row=12, column=0, sticky='nw', pady=5)
    sf_sweep_frame.grid(
        row=12, column=1, sticky='ne', pady=5)
    self.sf_sweep_use_tk = tk.IntVar()
    sf_sweep_button1 = tk.Radiobutton(sf_sweep_frame,
                                      text="yes",
                                      variable=self.sf_sweep_use_tk,
                                      command=lambda fr=self.sf_sweep_use_tk: set_sf_sweep_use(
                                          fr),
                                      value=1, font=widgetfont)
    sf_sweep_button2 = tk.Radiobutton(sf_sweep_frame,
                                      text="no",
                                      variable=self.sf_sweep_use_tk,
                                      command=lambda fr=self.sf_sweep_use_tk: set_sf_sweep_use(
                                          fr),
                                      value=2, font=widgetfont)
    sf_sweep_button2.select()

    sf_sweep_button1.grid(row=0, column=0)
    sf_sweep_button2.grid(row=0, column=1)

    spreading_factor_label = tk.Label(
        right_frame, text='Spreading factor (SF)', font=widgetfont)
    spreading_factor_input = tk.Scale(
        right_frame, from_=7, to=12, orient="horizontal", command=setsf)

    spreading_factor_label.grid(row=13, column=0, sticky='nw', pady=10)
    spreading_factor_input.grid(row=13, column=1, sticky='ne', pady=10)

    propagation_model_label = tk.Label(right_frame, text='Propagation model', font=widgetfont)
    propagation_model_box = ttk.Combobox(right_frame,
                                         values=["Free space path loss", "Hata", "Lee", "Cost 231", "Ericsson"], font=widgetfont)
    propagation_model_label.grid(row=14, column=0, sticky='nw', pady=10)
    propagation_model_box.grid(row=14, column=1, sticky='ne', pady=10)
    propagation_model_box.current(0)
    propagation_model_box.bind("<<ComboboxSelected>>", lambda event,
                               val=propagation_model_box: setpropagation_model(val, event))
    grid_size_label = tk.Label(right_frame, text='Grid scale %', font=widgetfont)
    grid_size_scale = tk.Scale(right_frame, from_=10, to=200,
                               orient="horizontal", resolution='10', command=setgridsize, font=widgetfont)
    grid_size_scale.set(100)

    grid_size_label.grid(row=15, column=0, sticky='nw', pady=10)
    grid_size_scale.grid(row=15, column=1, sticky='ne', pady=10)

    Status = Text(right_frame, height=3, width=50, state=DISABLED, font=widgetfont)
    Status.grid(row=18, column=0, columnspan=2, sticky='nesw', pady=10)

    enter_button = tk.Button(right_frame, text='Visualize', font=widgetfont,
                             command=lambda: placement_thread(mycanvas, Status,RSSI_Threshold))

    enter_button.grid(row=16, column=0, columnspan=2, sticky='nesw', pady=5)
    reset_button = tk.Button(
        right_frame, text='Reset All', command=lambda: reset_all(), font=widgetfont)

    reset_button.grid(row=17, column=0, columnspan=2, sticky='nesw', pady=5)


class RSCanvas(Canvas):
    def __init__(self, parent, **kwargs):
        Canvas.__init__(self, parent, **kwargs)
        self.height = self.winfo_reqheight()
        self.width = self.winfo_reqwidth() - 350
        self.bind("<Configure>", self.resize)

    def resize(self, event):
        self.config(width=event.width - 350, height=event.height)
# https://stackoverflow.com/a/22837522


class Main(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        main_container = tk.Frame(self)

        main_container.grid(column=0, row=0, sticky="nsew")
        main_container.grid_rowconfigure(0, weight=1)
        main_container.grid_columnconfigure(0, weight=1)
        self.title("LoRaPlan")
        create_widgets(self, main_container)
        if("Win" in platform.system()):
            self.wm_state('zoomed')
        elif ("Linux" in platform.system()):
            self.wm_attributes('-zoomed', 1)

gwplanning = Main()
gwplanning.geometry("1400x820")
gwplanning.mainloop()
