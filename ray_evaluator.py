import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib import pyplot
from shapely.geometry import LineString, Point
import copy

from rangefinder import Rangefinder
from tsdf_inserter import ScanNormalTSDFRangeInserter
from tsdf import TSDF

def generateEnvironment():
    obstacle = LineString([(1, 1.5), (9, 1.5)])
    #x, y = obstacle.xy
    #ax.plot(x, y, color='black',  linewidth=1)
    return obstacle


def plotTSDFWithScan(tsdf, hits, sensor_origin, environment):    
    fig = plt.figure()
    ax = plt.subplot(311)
    extent = -tsdf.resolution/2.0, tsdf.size+tsdf.resolution/2.0, -tsdf.resolution/2.0, tsdf.size+tsdf.resolution/2.0
    plt.imshow(tsdf.tsdf, interpolation='nearest',extent=extent, cmap= 'seismic', origin="lower",vmin = -1, vmax = 1)
    plt.colorbar()
    
    environment_x, environment_y = environment.xy
    ax.plot(environment_x, environment_y, color='black',  linewidth=1)
    x_val = [x[0] for x in hits]
    y_val = [x[1] for x in hits]
    ax.scatter(x_val, y_val, marker='x')
    ax.scatter(sensor_origin[0], sensor_origin[1], marker='x')
    
    ax = plt.subplot(312)
    extent = -tsdf.resolution/2.0, tsdf.size+tsdf.resolution/2.0, -tsdf.resolution/2.0, tsdf.size+tsdf.resolution/2.0
    plt.imshow(tsdf.weights, interpolation='nearest',extent=extent, cmap= 'jet', origin="lower")
    plt.colorbar()
    
    ax = plt.subplot(313)
    extent = -tsdf.resolution/2.0, tsdf.size+tsdf.resolution/2.0, -tsdf.resolution/2.0, tsdf.size+tsdf.resolution/2.0
    tsdf_grund_truth = 0.*tsdf.tsdf
    it = np.nditer(tsdf_grund_truth, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = [[it.multi_index[0]],[it.multi_index[1]]]
        point_position = tsdf.getPositionAtCellIndex(idx)
        distance = environment.distance(Point(point_position))
        it[0] = min(distance, tsdf.truncation_distance)
        it.iternext()
    tsdf_error = 0.*tsdf.tsdf
    tsdf_error[tsdf.weights > 0] = np.abs(tsdf_grund_truth[tsdf.weights > 0] - np.abs(tsdf.tsdf[tsdf.weights>0]))
    #tsdf_error = np.abs(tsdf_error - np.abs(tsdf.tsdf))
    plt.imshow(tsdf_error, interpolation='nearest',extent=extent, cmap= 'jet', origin="lower",vmin =0, vmax = 0.2)
    plt.colorbar()

def plotTSDFErrorDeltas(tsdfs, hits, sensor_origin, environment):    
    n_tsdfs = len(tsdfs)
    for i in range(0, n_tsdfs-1):
        fig = plt.figure()
        ax = plt.subplot(111)
        extent = -tsdfs[0].resolution/2.0, tsdfs[0].size+tsdfs[0].resolution/2.0, -tsdfs[0].resolution/2.0, tsdfs[0].size+tsdfs[0].resolution/2.0
        tsdf_grund_truth = 0.*tsdfs[0].tsdf
        it = np.nditer(tsdf_grund_truth, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = [[it.multi_index[0]],[it.multi_index[1]]]
            point_position = tsdfs[i].getPositionAtCellIndex(idx)
            distance = environment.distance(Point(point_position))
            it[0] = min(distance, tsdfs[i].truncation_distance)
            it.iternext()
        tsdf_error_before = 0.*tsdfs[i].tsdf
        tsdf_error_before[tsdfs[i].weights > 0] = np.abs(tsdf_grund_truth[tsdfs[i].weights > 0] - np.abs(tsdfs[i].tsdf[tsdfs[i].weights>0]))
        tsdf_error_after = 0.*tsdfs[i+1].tsdf
        tsdf_error_after[tsdfs[i+1].weights > 0] = np.abs(tsdf_grund_truth[tsdfs[i+1].weights > 0] - np.abs(tsdfs[i+1].tsdf[tsdfs[i+1].weights>0]))
        plt.imshow(tsdf_error_after-tsdf_error_before, interpolation='nearest',extent=extent, cmap='seismic', origin="lower",vmin =-0.1, vmax = 0.1)
        plt.colorbar()
        

def singleRay(range_inserter):
    environment = generateEnvironment()
    rangefinder = Rangefinder()  
    tsdf = TSDF()
    
    tsdfs = []
    
    for x in range(1,10,1):
        sensor_origin = (x,1)    
        hits = rangefinder.scan(environment, sensor_origin)  
        range_inserter.insertScan(tsdf, hits, sensor_origin)   
        if x%2==0:
            plotTSDFWithScan(tsdf, hits, sensor_origin, environment)
            tsdfs += [copy.deepcopy(tsdf)]
    #plotTSDFErrorDeltas(tsdfs, hits, sensor_origin, environment)
    plt.show()
   
if __name__ == '__main__':
    default_inserter = ScanNormalTSDFRangeInserter();
    singleRay(default_inserter)
