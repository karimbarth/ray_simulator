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
    
    #Line
    #obstacle = LineString([(1, 1.5), (9, 2.3)])
    
    #slightly jagged line
    #obstacle = LineString([(1, 1.5), (3.2, 2.3), (4.8, 1.7), (7.1, 2.4), (9, 1.4)])
    
    #strongly jagged line
    #obstacle = LineString([(1, 1.5), (2.2, 2.3), (2.9, 1.7), (4.1, 2.4), (5.1, 1.4), (5.9, 2.3), (6.8, 1.7), (7.1, 2.4), (7.7, 1.4), (9, 2.4)])
    
    # sine wave
    '''
    sine_x = np.arange(1.0, 9.0, 0.1)
    sine_y = np.sin(sine_x*3)*0.25+1.75
    sine_points = []
    for i in range(1,len(sine_x)):
        sine_points += [[sine_x[i], sine_y[i]]]
    obstacle = LineString(sine_points)
    '''
    
    x = np.arange(1.0, 9.0, )
    points = []
    px = 1
    py = 1.75
    dx = 1
    dy = 0.5
    i_line = 1
    while px<9:
        step_type = i_line%4
        if step_type == 0 or step_type == 2:
            px += dx
        elif step_type == 1:
            py += dy
        elif step_type == 3:
            py -= dy 
        i_line += 1
        points += [[px, py]]
    obstacle = LineString(points)
        
    
    
    #x, y = obstacle.xy
    #ax.plot(x, y, color='black',  linewidth=1)
    return obstacle


def plotTSDFWithScan(tsdf, hits, sensor_origin, environment, caption):    
    fig = plt.figure()
    ax = plt.subplot(311)
    extent = -tsdf.resolution/2.0, tsdf.size+tsdf.resolution/2.0, -tsdf.resolution/2.0, tsdf.size+tsdf.resolution/2.0
    plt.imshow(tsdf.tsdf, interpolation='nearest',extent=extent, cmap= 'seismic', origin="lower",vmin = -1, vmax = 1)
    plt.ylim(0.5,2.5)
    plt.title('TSDF')
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
    plt.ylim(0.5,2.5)
    plt.title('TSDF Weights')
    plt.colorbar()
    
    '''
    ax.plot(environment_x, environment_y, color='black',  linewidth=1)    
    for hit in hits:
        hit_ray = np.transpose(np.array([hit, sensor_origin]))
        ax.plot(hit_ray[0], hit_ray[1], color='black',  linewidth=1)
    '''   
    
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
    plt.imshow(tsdf_error, interpolation='nearest',extent=extent, cmap= 'jet', origin="lower",vmin =0, vmax = 0.1)
    plt.ylim(0.5,2.5)
    plt.title('TSDF Error')
    plt.suptitle(caption)
    plt.colorbar()

def plotTSDFErrorDeltas(tsdfs, environment):    
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
        plt.ylim(0.5,2.5)        
        plt.colorbar()
        

def plotTSDFErrorDeltasMatrix(tsdfs, tsdf_identifiers, environment):    
    n_tsdfs = len(tsdfs)
    #fig = plt.figure()
    f, axarr = plt.subplots(n_tsdfs, n_tsdfs, sharex=True, sharey=True)

    sc = None
    for i in range(0, n_tsdfs):
        for j in range(0, n_tsdfs):
            ax = axarr[i,j] 
            if(i==n_tsdfs-1):                   
                ax.set_xlabel(tsdf_identifiers[j])
            if(j==0):                   
                ax.set_ylabel(tsdf_identifiers[i])
            #plt.subplot(n_tsdfs,n_tsdfs,(i+1) + j*n_tsdfs)
            extent = -tsdfs[0].resolution / 2.0, tsdfs[0].size+tsdfs[0].resolution/2.0, -tsdfs[0].resolution/2.0, tsdfs[0].size+tsdfs[0].resolution / 2.0
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
            tsdf_error_after = 0.*tsdfs[j].tsdf
            tsdf_error_after[tsdfs[j].weights > 0] = np.abs(tsdf_grund_truth[tsdfs[j].weights > 0] - np.abs(tsdfs[j].tsdf[tsdfs[j].weights>0]))
            sc = ax.imshow(tsdf_error_after-tsdf_error_before, interpolation='nearest',extent=extent, cmap='seismic', origin="lower",vmin =-0.1, vmax = 0.1)
            ax.set_ylim(0.5,2.5)     
    #Combined colorbar to the right
    f.subplots_adjust(right=0.8)
    cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
    f.colorbar(sc, cax=cbar_ax)
    #f.colorbar(sc)
    
def plotTSDFErrorMatrix(tsdfs, tsdf_identifiers, environment):    
    n_tsdfs = len(tsdfs)
    #fig = plt.figure()
    f, axarr = plt.subplots(n_tsdfs,  sharex=True, sharey=True)

    sc = None
    for i in range(0, n_tsdfs):
        ax = axarr[i]                 
        ax.set_xlabel(tsdf_identifiers[i])
        #plt.subplot(n_tsdfs,n_tsdfs,(i+1) + j*n_tsdfs)
        extent = -tsdfs[0].resolution / 2.0, tsdfs[0].size+tsdfs[0].resolution/2.0, -tsdfs[0].resolution/2.0, tsdfs[0].size+tsdfs[0].resolution / 2.0
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
        sc = ax.imshow(tsdf_error_before, interpolation='nearest',extent=extent, cmap='jet', origin="lower",vmin =0., vmax = 0.1)
        ax.set_ylim(0.5,2.5)     
    #Combined colorbar to the right
    f.subplots_adjust(right=0.8)
    cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
    f.colorbar(sc, cax=cbar_ax)
    #f.colorbar(sc)

def singleRay(range_inserter, environment, title_trunk='Default'):
    rangefinder = Rangefinder()  
    tsdf = TSDF()
    
    tsdfs = []
    hits = None
    sensor_origin = None
    for x in range(22,25,1):
        sensor_origin = (x/5.0,1.0)    
        hits = rangefinder.scan(environment, sensor_origin)  
        range_inserter.insertScan(tsdf, hits, sensor_origin)   
        #if x==100:
        #    plotTSDFWithScan(tsdf, hits, sensor_origin, environment, title_trunk + ' TSDF')
            #tsdfs += [copy.deepcopy(tsdf)]
            
    plotTSDFWithScan(tsdf, hits, sensor_origin, environment, title_trunk + ' TSDF')
    #plotTSDFErrorDeltas(tsdfs, hits, sensor_origin, environment)
    return tsdf
   
if __name__ == '__main__':        
    environment = generateEnvironment()
    tsdfs = []
    tsdf_identifiers = []
    
    default_inserter = ScanNormalTSDFRangeInserter(use_normals_weight=False);
    tsdf_default = singleRay(default_inserter, environment)
    tsdfs += [tsdf_default]
    tsdf_identifiers += ['const']
    '''
    default_inserter = ScanNormalTSDFRangeInserter(use_normals_weight=True, n_normal_samples=8);
    tsdf_weights = singleRay(default_inserter, environment, 'Weight=cos(alpha) ')
    tsdfs += [tsdf_weights]
    tsdf_identifiers += ['angle']    
    
    default_inserter = ScanNormalTSDFRangeInserter(use_normals_weight=False, n_normal_samples=8, use_distance_cell_to_observation_weight=True, use_distance_cell_to_ray_weight=False);
    tsdfs += [singleRay(default_inserter, environment, 'Weight=distance1 ')]
    tsdf_identifiers += ['dist1']       
    
    default_inserter = ScanNormalTSDFRangeInserter(use_normals_weight=False, n_normal_samples=8, use_distance_cell_to_observation_weight=False, use_distance_cell_to_ray_weight=True);
    tsdfs += [singleRay(default_inserter, environment, 'Weight=distance2 ')]
    tsdf_identifiers += ['dist2']    
    
    default_inserter = ScanNormalTSDFRangeInserter(use_normals_weight=True, n_normal_samples=8, use_distance_cell_to_observation_weight=True);
    tsdf_weights_distance_scaled = singleRay(default_inserter, environment, 'Weight=cos(alpha)*distance ')
    tsdfs += [tsdf_weights_distance_scaled]
    tsdf_identifiers += ['angle*dist1']    
    '''
    
    default_inserter = ScanNormalTSDFRangeInserter(use_normals_weight=False, n_normal_samples=8, use_scale_distance=True, use_distance_cell_to_ray_weight=False);
    tsdfs += [singleRay(default_inserter, environment, 'Weight=cost scale_dist=true ')]
    tsdf_identifiers += ['const + scale dist']    
    
    default_inserter = ScanNormalTSDFRangeInserter(use_normals_weight=True, n_normal_samples=8, use_distance_cell_to_observation_weight=True, use_distance_cell_to_ray_weight=False,  use_scale_distance=True);
    tsdfs += [singleRay(default_inserter, environment, 'Weight=cos(alpha)*distance1 + scale_dist ')]
    tsdf_identifiers += ['angle*dist1 + scale_dist']
    '''
    default_inserter = ScanNormalTSDFRangeInserter(use_normals_weight=True, n_normal_samples=8, use_distance_cell_to_observation_weight=True, use_distance_cell_to_ray_weight=True,  use_scale_distance=True);
    tsdfs += [singleRay(default_inserter, environment, 'Weight=cos(alpha)*distance1*distance2  + scale_dist  ')]
    tsdf_identifiers += ['angle*dist1*dist2 + scale_dist']
'''
    
    
    plotTSDFErrorDeltasMatrix(tsdfs, tsdf_identifiers, environment)
    plotTSDFErrorMatrix(tsdfs, tsdf_identifiers, environment)
    plt.show()
