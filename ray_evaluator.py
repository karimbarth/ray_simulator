import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib import pyplot
from shapely.geometry import LineString

from rangefinder import Rangefinder
from tsdf_inserter import DefaultTSDFRangeInserter, ScanNormalTSDFRangeInserter
from tsdf import TSDF

def generateEnvironment():
    obstacle = LineString([(1, 1.5), (9, 1.5)])
    #x, y = obstacle.xy
    #ax.plot(x, y, color='black',  linewidth=1)
    return obstacle


def plotTSDFWithScan(tsdf, hits, sensor_origin, environment):    
    fig = plt.figure()
    ax = plt.subplot(211)
    extent = -tsdf.resolution/2.0, tsdf.size+tsdf.resolution/2.0, -tsdf.resolution/2.0, tsdf.size+tsdf.resolution/2.0
    plt.imshow(tsdf.tsdf, interpolation='nearest',extent=extent, cmap= 'seismic', origin="lower",vmin = -1, vmax = 1)
    
    environment_x, environment_y = environment.xy
    ax.plot(environment_x, environment_y, color='black',  linewidth=1)
    x_val = [x[0] for x in hits]
    y_val = [x[1] for x in hits]
    ax.scatter(x_val, y_val, marker='x')
    ax.scatter(sensor_origin[0], sensor_origin[1], marker='x')
    
    ax = plt.subplot(212)
    extent = -tsdf.resolution/2.0, tsdf.size+tsdf.resolution/2.0, -tsdf.resolution/2.0, tsdf.size+tsdf.resolution/2.0
    plt.imshow(tsdf.weights, interpolation='nearest',extent=extent, cmap= 'jet', origin="lower")
    

def singleRay(range_inserter):
    environment = generateEnvironment()
    rangefinder = Rangefinder()  
    tsdf = TSDF()
    
    for x in range(1,10,1):
        sensor_origin = (x,1)    
        print(sensor_origin)
        hits = rangefinder.scan(environment, sensor_origin)  
        print(len(hits))
        range_inserter.insertScan(tsdf, hits, sensor_origin)    
        plotTSDFWithScan(tsdf, hits, sensor_origin, environment)   
    plt.show()
   
if __name__ == '__main__':
    default_inserter = ScanNormalTSDFRangeInserter();
    singleRay(default_inserter)