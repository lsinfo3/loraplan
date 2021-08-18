import os
import math
import sys
import re
import pandas as pd
import numpy
from PIL import Image
import pandas as pd
from random import gauss
import matplotlib.path

# https://de.wikipedia.org/wiki/Mercator-Projektion

centerlon = 0#the center of the mercator projection
minx = sys.maxsize#minimal x value
miny = sys.maxsize#minimal y value
scalefactor_g = 0#global scalefactor
lastdemname = ""#used to keep previous dem file loaded to improve performance
lastdem = 0#last dems data (will be overwritten with a dataframe during execution)
fresnel_use = True#set by ui
last_1m_file = 0#only used by the 1m dem not a general option
last_1m_name = ""#only used by the 1m dem not a general option
raycheckdistance = 1#used for kml collision calculation
allkmlcoordinates = 0#used for kml collision calculation
kml_file_path = ""#set by ui
kml_max_height = 0#set by software to match the kml input
kml_polys = 0#will be overwritten with the actual kml data
use_1m_dem = False#set by files in the DEM directory
extend = 1201#extend of the SRTM DEM (set by software)


def set_fresnel_use(fres):
    global fresnel_use
    fresnel_use = fres

#mercator projection used by the map
def mercator_projection(lat, lon, centerlon):
    coordxy = [0, 0]
    coordxy[0] = (math.radians(lon)-math.radians(centerlon))*5000000
    # x is scaled by 4 to unsquish the map
    # x and y scaled by 1000000 to make 100m = 10 units
    coordxy[1] = math.log(
        math.tan(math.radians(lat))+math.sqrt(math.pow(math.tan(math.radians(lat)), 2)+1))*-5000000  # Minus, um Nord/Süd umzudrehen
    return coordxy
#inverse of the mercator projection
def gudermann(x, y, centerlon):
    x = x/float(5000000)
    y = y/float(-5000000)
    coordlatlon = [0, 0]
    coordlatlon[0] = math.degrees(math.atan(math.sinh(y)))
    coordlatlon[1] = math.degrees(x+math.radians(centerlon))
    return coordlatlon

#gets the center coordinate for the projection
def findcenterlon(dataframe, loncolumn):
    global centerlon
    minlon = sys.maxsize
    maxlon = 0
    for index, row in dataframe.iterrows():
        if(row[loncolumn] < minlon):
            minlon = row[loncolumn]
        elif(row[loncolumn] > maxlon):
            maxlon = row[loncolumn]
    centerlon = minlon + 0.5*(maxlon-minlon)
    return centerlon

#generates the map drawn to the screen with the scaling applied
def generatescreenmap(dataframe, loncolumn, latcolumn, scalefactor):
    global minx
    global miny
    global scalefactor_g
    scalefactor_g = scalefactor
    mapdataframe = generatemapdataframe(dataframe, loncolumn, latcolumn)
    miny = sys.maxsize
    maxy = 0
    minx = sys.maxsize
    maxx = 0
    for index, row in mapdataframe.iterrows():
        if(row[0] < minx):
            minx = row[0]
        elif(row[0] > maxx):
            maxx = row[0]
        if(row[1] < miny):
            miny = row[1]
        elif(row[1] > maxy):
            maxy = row[1]
    coordinates = []
    for index, row in mapdataframe.iterrows():
        coordinates.append([((row[0]-minx)*scalefactor),
                            ((row[1]-miny)*scalefactor)])
    coorddataframe = pd.DataFrame(coordinates)
    return coorddataframe
# only accounts for y to not stretch the map and y is the limitation for most screens

#generates the map for the calculations
def generatemapdataframe(dataframe, loncolumn, latcolumn):
    coordinates = []
    centerlon = findcenterlon(dataframe, loncolumn)
    for index, row in dataframe.iterrows():
        coordinates.append(mercator_projection(
            row[latcolumn], row[loncolumn], centerlon))
    coorddataframe = pd.DataFrame(coordinates)
    return coorddataframe
#0 is x
#1 is y

#returns the elevation from coordinates on the map dataframe
def getelevation_from_map(gdem_file_path, x, y):
    global centerlon
    global minx
    global miny
    global centerlon
    scalefactor = (1/100)
    mx = (x/scalefactor)+minx
    my = (y/scalefactor)+miny
    latlon = gudermann(mx, my, centerlon)
    return getelevation(gdem_file_path, latlon[0], latlon[1])

#returns the elevation from coordinate pairs
def getelevation(gdem_file_path, lat, lon):
    # requires a SRTMv3 file with the matching lat,lon in the specified file path.
    # changes x,y koord to lat lon: #1200 for 3 arc sec Modell
    # Lat=N-Teil+x/1200
    # Lon=E-Teil+y/1200

    global lastdemname
    global lastdem
    global raycheckdistance
    global use_1m_dem
    global extend
    if(os.path.exists(gdem_file_path+"/vermessungsamt_bayern_1m.tiff") and os.path.exists(gdem_file_path+"/vermessungsamt_bayern_1m.csv")):
        try:
            elev = get_elevation_from_1m_dem(gdem_file_path, lat, lon)
            use_1m_dem = True
            raycheckdistance = 0.3
        except:
            elev = 0
    if(not use_1m_dem):
        try:
            # extend = 1201  # for SRTM3
            if(lat >= 0):
                gdem_file_open = gdem_file_path + \
                    "/N"+str("%02d" % math.floor(lat))
            else:
                gdem_file_open = gdem_file_path + \
                    "/S"+str("%02d" % math.floor(lat))
            if(lon >= 0):
                gdem_file_open += "E"+str("%03d" % math.floor(lon))+".hgt"
            else:
                gdem_file_open += "W"+str("%03d" % math.floor(lon))+".hgt"
            if(lastdemname == gdem_file_open):
                gdem_file = lastdem
            else:
                height_file = open(gdem_file_open, "rb")
                lastdemname = gdem_file_open
                gdem_file = numpy.fromfile(
                    height_file, numpy.dtype('>i2'), -1)
                extend = int(math.sqrt(len(gdem_file)))
                if(extend > 1201):
                    raycheckdistance = 0.3
                else:
                    raycheckdistance = 0.9
                gdem_file = numpy.reshape(gdem_file, (extend, extend))
                lastdem = gdem_file
            x = extend - int((lat-int(lat))*(extend-1))
            y = int((lon-int(lon))*(extend-1))
            elev = int(gdem_file[x][y])
        except:
            elev = 0
    return(elev)

#used to obtain elevation data from a specific 1m DEM
def get_elevation_from_1m_dem(gdem_file_path, lat, lon):
    global last_1m_file
    global last_1m_name
    Image.MAX_IMAGE_PIXELS = None
    if(last_1m_name != (gdem_file_path+"/vermessungsamt_bayern_1m.tiff")):
        im = Image.open(gdem_file_path+"/vermessungsamt_bayern_1m.tiff")
        last_1m_file = im
        last_1m_name = gdem_file_path+"/vermessungsamt_bayern_1m.tiff"
    else:
        im = last_1m_file
    setupfile = pd.read_csv(
        gdem_file_path+"/vermessungsamt_bayern_1m.csv", sep=',')
    pixely = int((lon-setupfile.iloc[0][0])/setupfile.iloc[0][2])
    pixelx = int((lat-setupfile.iloc[0][1])/setupfile.iloc[0][3])
    cordinate = pixelx, pixely
    return(im.getpixel(cordinate))


def set_kml_file_path(path_to_set):
    global kml_file_path
    kml_file_path = path_to_set


def readkml():
    # Reads all Coordinates in the KML Files in a single Directory
    # Output: List of Polygons defined by their Coordinates (separated by space)
    global allkmlcoordinates
    global kml_file_path
    allkmlcoordinates = []
    if(kml_file_path != ""):
        for currfile in os.listdir(kml_file_path):
            if currfile.endswith(".kml"):
                with open(os.path.join(kml_file_path, currfile), 'r') as myfile:
                    kml = myfile.read()
                    coords = re.compile(r'<coordinates>(.*)</coordinates>')
                    matches = re.findall(coords, kml)
                    for match in matches:
                        allkmlcoordinates.append(match)
# https://matplotlib.org/3.1.1/api/path_api.html
# returns true if any polygon in the given directory overlaps with sightline
# set_kml_file_path has to be set beforehand

#ray check if any kml polygon presents an obstacle to the line of sight
def kmlintersectswithline(lat_1, lon_1, lat_2, lon_2):
    global allkmlcoordinates
    global kml_max_height
    global centerlon
    global kml_polys
    lowlat = 0
    highlat = 0
    lowlon = 0
    highlon = 0
    if lat_1 < lat_2:
        lowlat = lat_1
        highlat = lat_2
    else:
        lowlat = lat_2
        highlat = lat_1
    if lon_1 < lon_2:
        lowlon = lon_1
        highlon = lon_2
    else:
        lowlon = lon_2
        highlon = lon_1

    if(allkmlcoordinates == 0):
        readkml()
    if(kml_polys == 0):
        polys = []
        for coords in allkmlcoordinates:
            points = []
            singlepoint = coords.split(' ')
            for coord in singlepoint:
                coord = coord.split(',')
                currpoint = []
                for toappend in coord:
                    try:
                        currpoint.append(float(toappend))
                    except:
                        currpoint.append(0.0)
                if(currpoint[2] > kml_max_height):
                    kml_max_height = currpoint[2]
                points.append(currpoint)
            polys.append(points)
        kml_polys = polys
    # points contains all coordinates from the currently evaluated polygon.
    toreturn = []
    for polygon in kml_polys:
        if(lowlon < polygon[0][0] and polygon[0][0] < highlon and lowlat < polygon[0][1] and polygon[0][1] < highlat):
            bound = []
            for points in polygon:
                bound.append([points[0], points[1]])
            path = matplotlib.path.Path(bound)
            line = matplotlib.path.Path(numpy.array(
                [[lon_1, lat_1], [lon_2, lat_2]]), closed=False)
            if(path.intersects_path(line)):
                toreturn.append(polygon)
    return toreturn

def ericsson(sensorgrid, gwx, gwy, sensorsize, gateway_file, sf):
    try:
        gw_height = int(gateway_file.iloc[int(sensorgrid[gwx][gwy][2])][3])
        gw_type = gateway_file.iloc[int(sensorgrid[gwx][gwy][2])][4]
    except:
        gw_height = 15
        gw_type = "urban"
        sens_height = 1
    a_0 = 36.2
    a_1 = 30.2
    a_2 = 12.0
    a_3 = 0.1
    if(gw_type == "urban"):
        a_0 = 36.2
        a_1 = 30.2
        a_2 = 12.0
        a_3 = 0.1
    elif(gw_type == "suburban"):
        a_0 = 43.20
        a_1 = 68.93
        a_2 = 12.0
        a_3 = 0.1
    elif(gw_type == "rural"):
        a_0 = 45.95
        a_1 = 100.6
        a_2 = 12.0
        a_3 = 10.0
    sens_height = 1
    # represents Spreading Factor Tolerances for Path Loss in dB
    plrange = [131, 134, 137, 140, 141, 144]
    distance = 10**((a_0+a_2*math.log10(gw_height)-plrange[sf-7]-3.2*(
        math.log10(11.75*sens_height)**2)+89.460750)/(-a_3*math.log10(gw_height)-a_1))
    distance = distance*10  #Change to 100m grid
    return distance


def hata(sensorgrid, gwx, gwy, sensorsize, gateway_file, sf):
    try:
        gw_height = int(gateway_file.iloc[int(sensorgrid[gwx][gwy][2])][3])
        gw_type = gateway_file.iloc[int(sensorgrid[gwx][gwy][2])][4]
        sens_height = 1
    except:
        gw_height = 15
        gw_type = "urban"
        sens_height = 1
    # represents Spreading Factor Tolerances for Path Loss in dB
    plrange = [131, 134, 137, 140, 141, 144]
    distance = 10**(-(69.55+76.872985-13.82*math.log10(gw_height)-3.2*(math.log10(
        11.75*sens_height)**2)+4.97-plrange[sf-7])/(44.9-6.55*math.log10(gw_height)))
    distance = distance*10
    return distance


def cost231(sensorgrid, gwx, gwy, sensorsize, gateway_file, sf):
    try:
        gw_height = int(gateway_file.iloc[int(sensorgrid[gwx][gwy][2])][3])
        gw_type = gateway_file.iloc[int(sensorgrid[gwx][gwy][2])][4]
        sens_height = 1
    except:
        gw_height = 15
        gw_type = "urban"
        sens_height = 1
    c_m = 0
    a_h = 3.2*(math.log10(11.75*sens_height))**2 - 4.79
    if(gw_type == "urban"):
        pass
    elif(gw_type == "suburban"):
        c_m = 3
        a_h = (1.1*math.log10(868.1)-0.7) * \
            sens_height-(1.5*math.log10(868.1)-0.8)
    elif(gw_type == "rural"):
        c_m = 3
        a_h = (1.1*math.log10(868.1)-0.7) * \
            sens_height-(1.5*math.log10(868.1)-0.8)
    # represents Spreading Factor Tolerances for Path Loss in dB
    plrange = [131, 134, 137, 140, 141, 144]
    distance = 10**((46.3+33.9*math.log10(868.1)-13.82*math.log10(gw_height) -
                     a_h-plrange[sf-7]+c_m)/-(44.9-6.55*math.log10(gw_height)))
    distance = distance*10
    return distance


def lee(sensorgrid, gwx, gwy, sensorsize, gateway_file, sf):
    try:
        gw_height = int(gateway_file.iloc[int(sensorgrid[gwx][gwy][2])][3])
        gw_type = gateway_file.iloc[int(sensorgrid[gwx][gwy][2])][4]
        sens_height = 1
    except:
        gw_height = 15
        gw_type = "urban"
        sens_height = 1
    # represents Spreading Factor Tolerances for Path Loss in dB
    plrange = [131, 134, 137, 140, 141, 144]
    L_0_opt = [89, 101.7, 110]
    g_opt = [43.5, 38.5, 36.8]
    L_0 = 0
    g = 0
    F_1 = (gw_height/30.48)**2
    F_2 = (1/2)
    if(sens_height > 3):
        F_3 = (sens_height/3)**2
    else:
        F_3 = (sens_height/3)
    F_4 = (868.1/900)**(-2.5)
    F_5 = 2
    if(gw_type == "urban"):
        L_0 = L_0_opt[2]
        g = g_opt[2]
    elif(gw_type == "suburban"):
        L_0 = L_0_opt[1]
        g = g_opt[1]
    elif(gw_type == "rural"):
        L_0 = L_0_opt[0]
        g = g_opt[0]
    F_0 = F_1*F_2*F_3*F_4*F_5
    distance = 10**((L_0-plrange[sf-7]-10*math.log10(F_0))/(-g))
    distance = distance*10
    return distance

#unfinished implementation of a specific adapted model
def cost231_LoRa(sensorgrid, gwx, gwy, sensorsize, gateway_file, gateway_file_grid, sf):
    try:
        gw_height = int(gateway_file.iloc[int(sensorgrid[gwx][gwy][2])][3])
        gw_type = gateway_file.iloc[int(sensorgrid[gwx][gwy][2])][4]
        sens_height = 1
        plrange = [131, 134, 137, 140, 141, 144]
        num_gws = len(gateway_file)
        G_0 = gauss(0, 8)
        L_s = 0
        for i in range(0, num_gws):
            x = gateway_file_grid.iloc[i][1]
            y = gateway_file_grid.iloc[i][2]
            distance = math.sqrt(math.pow(abs(x-gwx), 2) +
                                 math.pow(abs(y-gwy), 2))
            Rxy = 2**(-distance/0.3679)
            G_i = gauss(0, 8)
            L_i = math.sqrt(Rxy)*G_0+math.sqrt(1-Rxy)*G_i
            L_s += L_i
        distance = 10**((plrange[sf-7]+18*math.log10(sens_height) +
                         21*math.log10(868.1)+80+L_s)/(40*(1-4*10**(-3)*gw_height)))
    except:
        distance = 10
    return distance

#rssi calculation (uses the original unmodified PL models)
def calculaterssi(sensorgrid, propmodel, sf, distance, gwx, gwy, gateway_file):
    if(distance == 0):
        distance = 0.1
    try:
        gw_height = int(gateway_file.iloc[int(sensorgrid[gwx][gwy][2])][3])
        gw_type = gateway_file.iloc[int(sensorgrid[gwx][gwy][2])][4]
    except:
        gw_height = 15
        gw_type = "urban"
    distance = distance/10
    sens_height = 1
    pl = 0
    tx_out = 8  # dBm SX1257
    if(propmodel == "fspl"):
        pl = 20*math.log10(868.1)+20*math.log10(distance)+32.45
    elif(propmodel == "manhattan"):
        pl = 20*math.log10(868.1)+20*math.log10(distance)+32.45
    elif(propmodel == "ericsson"):
        a_0 = 36.2
        a_1 = 30.2
        a_2 = 12.0
        a_3 = 0.1
        if(gw_type == "urban"):
            a_0 = 36.2
            a_1 = 30.2
            a_2 = 12.0
            a_3 = 0.1
        elif(gw_type == "suburban"):
            a_0 = 43.20
            a_1 = 68.93
            a_2 = 12.0
            a_3 = 0.1
        elif(gw_type == "rural"):
            a_0 = 45.95
            a_1 = 100.6
            a_2 = 12.0
            a_3 = 10.0
        sens_height = 1
        pl = a_0+a_1*math.log10(distance)+a_2*math.log10(gw_height)+a_3*math.log10(gw_height)*math.log10(
            distance)-3.2*((math.log10(11.75*sens_height))**2)+89.460750
    elif(propmodel == "hata"):
        pl = 69.55+26.16*math.log10(868.1)-13.82*math.log10(gw_height)-3.2*((math.log10(
            11.75*sens_height))**2)+4.97+(44.9-6.55*math.log10(gw_height))*math.log10(distance)
    elif(propmodel == "cost231"):
        c_m = 0
        a_h = 3.2*((math.log10(11.75*sens_height))**2) - 4.79
        if(gw_type == "urban"):
            pass
        elif(gw_type == "suburban"):
            c_m = 3
            a_h = (1.1*math.log10(868.1)-0.7) * \
                sens_height-(1.5*math.log10(868.1)-0.8)
        elif(gw_type == "rural"):
            c_m = 3
            a_h = (1.1*math.log10(868.1)-0.7) * \
                sens_height-(1.5*math.log10(868.1)-0.8)
        pl = 46.3+33.9*math.log10(868.1)-13.82*math.log10(gw_height)-a_h+(
            44.9-6.55*math.log10(sens_height))*math.log10(distance)+c_m
    elif(propmodel == "lee"):
        L_0_opt = [89, 101.7, 110]
        g_opt = [43.5, 38.5, 36.8]
        L_0 = 0
        g = 0
        F_1 = (gw_height/30.48)**2
        F_2 = (1/2)
        if(sens_height > 3):
            F_3 = (sens_height/3)**2
        else:
            F_3 = (sens_height/3)
        F_4 = (868.1/900)**(-2.5)
        F_5 = 2
        if(gw_type == "urban"):
            L_0 = L_0_opt[2]
            g = g_opt[2]
        elif(gw_type == "suburban"):
            L_0 = L_0_opt[1]
            g = g_opt[1]
        elif(gw_type == "rural"):
            L_0 = L_0_opt[0]
            g = g_opt[0]
        F_0 = F_1*F_2*F_3*F_4*F_5
        pl = L_0+g*math.log10(distance)-10*math.log10(F_0)
    return (tx_out-pl)


def calculatereachability_with_model(sensorgrid, linkgrid, gdem_file_path, gwx, gwy, sensorsize, propmodel, gateway_file, sf):
    gwreach = 10
    if(propmodel == "ericsson"):
        gwreach = ericsson(sensorgrid, gwx, gwy, sensorsize, gateway_file, sf)
    elif(propmodel == "hata"):
        gwreach = hata(sensorgrid, gwx, gwy, sensorsize, gateway_file, sf)
    elif(propmodel == "cost231"):
        gwreach = cost231(sensorgrid, gwx, gwy, sensorsize, gateway_file, sf)
    elif(propmodel == "lee"):
        gwreach = lee(sensorgrid, gwx, gwy, sensorsize, gateway_file, sf)
    return calculatereachability(sensorgrid, linkgrid, gdem_file_path, gwx, gwy, sensorsize, gwreach, propmodel, gateway_file, sf)


def calculatereachability(sensorgrid, linkgrid, gdem_file_path, gwx, gwy, sensorsize, gwreach, propmodel, gateway_file, sf):
    if(gdem_file_path != ""):
        return calculatereachability_with_height(sensorgrid, linkgrid, gdem_file_path, gwx, gwy, sensorsize, gwreach, propmodel, gateway_file, sf)
    else:
        return calculatereachability_without_height(sensorgrid, linkgrid, gwx, gwy, sensorsize, gwreach, propmodel, gateway_file, sf)

#reachability calculation without a DEM or KML present
def calculatereachability_without_height(sensorgrid, linkgrid, gwx, gwy, sensorsize, gwreach, propmodel, gateway_file, sf):
    boundxmin = int(gwx-gwreach)
    boundxmax = int(gwx+gwreach)
    boundymin = int(gwy-gwreach)
    boundymax = int(gwy+gwreach)
    if(boundxmax > len(sensorgrid)):
        boundxmax = len(sensorgrid)
    if(boundymax > len(sensorgrid[0])):
        boundymax = len(sensorgrid[0])
    if(boundxmin < 0):
        boundxmin = 0
    if(boundymin < 0):
        boundymin = 0
    reachablesensors = []
    reachablesensors.append([gwx, gwy])
    sensorgrid[gwx][gwy][1] = 0
    for x in range(boundxmin, boundxmax):
        for y in range(boundymin, boundymax):
            numsensors = 0
            distance = math.sqrt(math.pow(abs(x-gwx), 2) +
                                 math.pow(abs(y-gwy), 2))
            if(distance <= gwreach):
                if(0 < x < len(sensorgrid) and 0 < y < len(sensorgrid[0])):
                    for i in range(0, sensorsize):
                        numsensors += sensorgrid[x][y][i+3]
                    if(numsensors > 0):
                        sensorgrid[gwx][gwy][1] += numsensors
                        reachablesensors.append([x, y])
                        rssi = calculaterssi(
                            sensorgrid, propmodel, sf, distance, gwx, gwy, gateway_file)
                        if(linkgrid[x][y][3] == [] or linkgrid[x][y][3][-1] != sensorgrid[gwx][gwy][2]):
                            linkgrid[x][y][2] += 1
                            linkgrid[x][y][3].append(
                                int(sensorgrid[gwx][gwy][2]))
                        if(linkgrid[x][y][4] > sf):
                            linkgrid[x][y][4] = sf
                            linkgrid[x][y][1] = rssi
                            linkgrid[x][y][0] = sensorgrid[gwx][gwy][2]
                        if(linkgrid[x][y][4] == sf):
                            if(rssi >= linkgrid[x][y][1]):
                                linkgrid[x][y][1] = rssi
                                linkgrid[x][y][0] = sensorgrid[gwx][gwy][2]
                            #print("rssi "+str(distance)+" ="+str(rssi)+" GWs: "+str(linkgrid[x][y][3]))

    return reachablesensors

#Reachability if a DEM is present
def calculatereachability_with_height(sensorgrid, linkgrid, gdem_file_path, gwx, gwy, sensorsize, gwreach, propmodel, gateway_file, sf):
    boundxmin = int(gwx-gwreach)
    boundxmax = int(gwx+gwreach)
    boundymin = int(gwy-gwreach)
    boundymax = int(gwy+gwreach)
    if(boundxmax > len(sensorgrid)):
        boundxmax = len(sensorgrid)
    if(boundymax > len(sensorgrid[0])):
        boundymax = len(sensorgrid[0])
    if(boundxmin < 0):
        boundxmin = 0
    if(boundymin < 0):
        boundymin = 0
    reachablesensors = []
    reachablesensors.append([gwx, gwy])
    sensorgrid[gwx][gwy][1] = 0
    for x in range(boundxmin, boundxmax):
        for y in range(boundymin, boundymax):
            numsensors = 0
            distance = math.sqrt(math.pow(abs(x-gwx), 2) +
                                 math.pow(abs(y-gwy), 2))
            if(distance <= gwreach):
                if(0 < x < len(sensorgrid) and 0 < y < len(sensorgrid[0])):
                    if(is_in_lineofsight(gdem_file_path, gwx, gwy, x, y)):
                        for i in range(0, sensorsize):
                            numsensors += sensorgrid[x][y][i+3]
                        if(numsensors > 0):
                            sensorgrid[gwx][gwy][1] += numsensors
                            reachablesensors.append([x, y])
                            rssi = calculaterssi(
                                sensorgrid, propmodel, sf, distance, gwx, gwy, gateway_file)
                            if(linkgrid[x][y][3] == [] or linkgrid[x][y][3][-1] != sensorgrid[gwx][gwy][2]):
                                linkgrid[x][y][2] += 1
                                linkgrid[x][y][3].append(
                                    int(sensorgrid[gwx][gwy][2]))
                            if(linkgrid[x][y][4] > sf):
                                linkgrid[x][y][4] = sf
                                linkgrid[x][y][1] = rssi
                                linkgrid[x][y][0] = sensorgrid[gwx][gwy][2]
                            if(linkgrid[x][y][4] == sf):
                                if(rssi >= linkgrid[x][y][1]):
                                    linkgrid[x][y][1] = rssi
                                    linkgrid[x][y][0] = sensorgrid[gwx][gwy][2]
    return reachablesensors

#line of sight calculation
def is_in_lineofsight(gdem_file_path, x_1, y_1, x_2, y_2):
    global fresnel_use
    global raycheckdistance
    global centerlon
    global kml_file_path
    global kml_max_height
    global minx
    global miny
    global scalefactor_g
    if(kml_file_path != "" and kml_max_height == 0):
        coords1 = gudermann(x_1, y_1, centerlon)
        kmlintersectswithline(coords1[0], coords1[1], coords1[0], coords1[1])
    dev_gw_distance = math.sqrt(math.pow(abs(x_1-x_2), 2) +
                                math.pow(abs(y_1-y_2), 2))
    is_straight = False  # normal situation
    if(x_1 == x_2 and y_1 == y_2):
        return True
    if(x_1 == x_2):
        is_straight = True  # line straight up
    isreachable = True
    highest_point = max(getelevation_from_map(
        gdem_file_path, x_1, y_1), getelevation_from_map(gdem_file_path, x_2, y_2))
    if(is_straight):
        for ny in range(y_1+1, y_2):
            nx = x_1
            distance = math.sqrt(math.pow(abs(x_1-nx), 2) +
                                 math.pow(abs(y_1-ny), 2))
            m_s = ((getelevation_from_map(gdem_file_path, x_1, y_1) -
                    getelevation_from_map(gdem_file_path, x_2, y_2))/(0-distance))
            t_s = getelevation_from_map(gdem_file_path, x_2, y_2)-m_s*distance
            maxh = (m_s*nx+t_s)/10
            if(fresnel_use):
                a = fresnel_radius(1, distance/10, dev_gw_distance/10)
                b = distance*1000
                c = math.sqrt(a**2+b**2)
                h = (a*b)/c
                maxh -= h
            if(getelevation_from_map(gdem_file_path, nx, ny) > maxh):
                isreachable = False
    else:
        m = ((y_1-y_2)/(x_1-x_2))
        t = y_2-m*x_2
        distance = math.sqrt(math.pow(abs(x_1-x_2), 2) +
                             math.pow(abs(y_1-y_2), 2))
        m_s = ((getelevation_from_map(gdem_file_path, x_1, y_1) -
                getelevation_from_map(gdem_file_path, x_2, y_2))/(0-distance))
        t_s = getelevation_from_map(gdem_file_path, x_2, y_2)-m_s*distance
        for nx in numpy.arange(x_1+1, x_2, raycheckdistance):
            ny = m*nx+t
            distance = math.sqrt(
                math.pow(abs(x_1-nx), 2) + math.pow(abs(y_1-ny), 2))
            maxh = (m_s*nx+t_s)/10
            if(fresnel_use):
                a = fresnel_radius(1, distance/10, dev_gw_distance/10)
                b = distance*1000
                c = math.sqrt(a**2+b**2)
                h = (a*b)/c
                maxh -= h
            if(getelevation_from_map(gdem_file_path, int(nx), int(ny)) > maxh):
                isreachable = False
    if(isreachable and kml_file_path != ""):
        if(x_1 != x_2 and y_1 != y_2):
            scalefactor = (1/100)
            xadj_1 = (x_1/scalefactor)+minx
            yadj_1 = (y_1/scalefactor)+miny
            xadj_2 = (x_2/scalefactor)+minx
            yadj_2 = (y_2/scalefactor)+miny
            latlon1 = gudermann(xadj_1, yadj_1, centerlon)
            latlon2 = gudermann(xadj_2, yadj_2, centerlon)
            polys_to_check = kmlintersectswithline(
                latlon1[0], latlon1[1], latlon2[0], latlon2[1])
            for polygon in polys_to_check:
                for points in polygon:
                    if(points[2] > 0):
                        m = ((yadj_1-yadj_2)/(xadj_1-xadj_2))
                        t = yadj_2-m*xadj_2
                        distance = math.sqrt(math.pow(abs(xadj_1-xadj_2), 2) +
                                             math.pow(abs(yadj_1-yadj_2), 2))
                        m_s = ((getelevation_from_map(gdem_file_path, xadj_1, yadj_1) -
                                getelevation_from_map(gdem_file_path, xadj_2, yadj_2))/(0-distance))
                        t_s = getelevation_from_map(
                            gdem_file_path, xadj_2, yadj_2)-m_s*distance
                        x = mercator_projection(
                            points[1], points[0], centerlon)[0]
                        maxh = (m_s*x+t_s)/10
                        if((getelevation(gdem_file_path, points[1], points[0])+points[2]) > maxh):
                            isreachable = False
    return isreachable


# n Fresnelzone number d1=distance from the end node to a specific point (in Km), receiver distance in Km
def fresnel_radius(n, d1, receiver_distance):
    F_n = math.sqrt((n*868.1*d1*(receiver_distance-d1))/(receiver_distance))
    return F_n

#Used for debuging and to present the value ranges
if __name__ == '__main__':
    print(mercator_projection(49.798727, 9.820321, 9.820321))
    print(mercator_projection(49.798727, 9.821715, 9.820321))
    print(mercator_projection(49.798727, 9.820321, 9.820321))
    print(mercator_projection(49.799704, 9.821715, 9.820321))
    print(mercator_projection(49.786708, 9.932243, 9.905249999999999))
    print(gudermann(2337.009552, 0, 9.905249999999999))
    print(gudermann(121.64944886400298, -5026279.68731216, 9.820321))
    print(fresnel_radius(1, 0.5, 1))
