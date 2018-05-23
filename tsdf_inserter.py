import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import math as math
from matplotlib import pyplot
from matplotlib import cm
from shapely.geometry import LineString

import tsdf


def getRaytracingHelperVariables(observation_origin, observation_ray,t_start, t_end, grid_size_inv):
    traversal_start = observation_origin + t_start * observation_ray
    traversal_end = observation_origin + t_end * observation_ray
    traversal_start_scaled = traversal_start * grid_size_inv
    traversal_end_scaled = traversal_end * grid_size_inv
    traversal_ray_scaled = traversal_end_scaled - traversal_start_scaled
    traversal_ray_scaled_inv =(1. / traversal_ray_scaled[0], 1. / traversal_ray_scaled[1])
    grid_index = np.round(traversal_start_scaled)
    grid_step = np.sign(traversal_ray_scaled)
    #adjustment = (grid_step > 0).astype(float)
    grid_index_adjusted = grid_index + 0.5*grid_step    
    t_max = (grid_index_adjusted - traversal_start_scaled) * traversal_ray_scaled_inv
    t_delta = grid_step * traversal_ray_scaled_inv
    return grid_index, grid_step, t_max, t_delta
'''
class DefaultTSDFRangeInserter:   
    
    def __init__(self):
        self.update_weight = 1
        
    def updateCell(self, tsdf, cell_index, update_distance, ray_length):       
        if(abs(update_distance)< tsdf.truncation_distance):
            updated_weight = tsdf.getWeight(cell_index) + self.update_weight
            updated_tsdf = (tsdf.getTSDF(cell_index) * tsdf.getWeight(cell_index) + update_distance * self.update_weight) / updated_weight    
            tsdf.setWeight(cell_index, updated_weight)
            tsdf.setTSDF(cell_index, updated_tsdf)
        
    def insertScan(self, tsdf, hits, origin):
        origin = np.array(origin)
        for hit in hits:
            hit = np.array(hit)
            grid_index, grid_step, t_max, t_delta = getRaytracingHelperVariables(origin, hit-origin, 0., 1.1, 1. / tsdf.resolution)
            t = 0
            ray = hit - origin
            ray_range = np.linalg.norm(ray)        
            range_inv = 1.0 / ray_range
            t_truncation_distance = tsdf.truncation_distance * range_inv
            t_start = 0.0
            t_end = 1.0 + t_truncation_distance
            while t < t_end :
                t_next = np.min(t_max)
                min_coeff_idx = np.argmin(t_max)
                sampling_point = origin + (t + t_next)/2 * ray
                cell_index = tsdf.getCellIndexAtPosition(sampling_point)
                cell_center = tsdf.getPositionAtCellIndex(cell_index)
                distance = np.linalg.norm(cell_center - origin)
                self.updateCell(tsdf, cell_index, ray_range - distance, ray_range)
                t = t_next
                t_max[min_coeff_idx] += t_delta[min_coeff_idx]
'''
#from  https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python   
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def angle(v):
    if(v.dot(np.array([0,1])) < 0):
        v = v
    return angle_between(v, np.array([0,1]))
                         

class ScanNormalTSDFRangeInserter:   
    
    def __init__(self):
        self.n_normal_samples = 8
        self.default_weight = 1
        self.normal_distance_factor = 1
        
        
    def updateCell(self, tsdf, cell_index, update_distance, ray_length, update_weight):       
        if(abs(update_distance)< tsdf.truncation_distance):
            updated_weight = min(tsdf.getWeight(cell_index) + update_weight,100)
            updated_tsdf = (tsdf.getTSDF(cell_index) * tsdf.getWeight(cell_index) + update_distance * update_weight) / (update_weight + tsdf.getWeight(cell_index))   
            tsdf.setWeight(cell_index, updated_weight)
            tsdf.setTSDF(cell_index, updated_tsdf)
        #tsdf.setWeight(cell_index, 0.5)
        #tsdf.setTSDF(cell_index, 0.5)
    
    
    def computeNormal(self, sample, neighbors):
        normals = []
        normal_distances = []
        normal_weights = []
        for neighbor in neighbors:
            tangent_angle = angle(sample-neighbor)
            normal_angle = tangent_angle - math.pi/2
            #print('normal_angle',normal_angle)
            normals += [normal_angle]
            normal_distance = np.linalg.norm(sample-neighbor)
            normal_distances += [normal_distance]
            normal_weights += [math.e**(-self.normal_distance_factor * normal_distance)]
            
        normals = np.array(normals)
        normal_weights = np.array(normal_weights)
        normal_mean = np.average(normals, weights=normal_weights)
        normal_var = np.average((normals-normal_mean)**2, weights=normal_weights)
        normal_weight_sum = np.sum(normal_weights)
        return normal_mean, normal_var, normal_weight_sum
        
    def drawScanWithNormals(self, hits, normal_orientations, sensor_origin, normal_weights):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x_val = [x[0] for x in hits]
        y_val = [x[1] for x in hits]
        ax.scatter(x_val, y_val, c=normal_weights, marker='x', cmap=cm.jet)
        ax.scatter(sensor_origin[0], sensor_origin[1], marker='x')
        for idx, normal_orientation in enumerate(normal_orientations):  
            normal_scale = 0.1
            dx = -normal_scale*np.sin(normal_orientation)
            dy = -normal_scale*np.cos(normal_orientation)        
            ax.arrow(x_val[idx], y_val[idx], dx, dy, fc='k', ec='k', color='b')
            ax.set_aspect('equal')
        #plt.show()
        
    def computeNormalBasedWeight(normal_orientation, normal_score, ray_orientation):
        pass
        
    
    def insertScan(self, tsdf, hits, origin):
        origin = np.array(origin)
        hits = np.array(hits)
        n_hits = len(hits)
        normal_orientations = []
        normal_orientation_variances = []
        normal_estimation_weight_sums = []
        for idx, hit in enumerate(hits):      
            #print('origin',origin)       
            #print('hit',hit)      
            '''
            neighbor_indices = np.array(list(range(idx-int(np.floor(self.n_normal_samples/2)), idx)) + list(range(idx+1, idx+int(np.ceil(self.n_normal_samples/2) + 1))))
            neighbor_indices = neighbor_indices[neighbor_indices >= 0]
            neighbor_indices = neighbor_indices[neighbor_indices < n_hits]
            normal_orientation, normal_var, normal_estimation_weight_sum = self.computeNormal(hit, hits[neighbor_indices])
            normal_orientations += [normal_orientation]
            normal_estimation_weight_sums += [normal_estimation_weight_sum]
            normal_orientation_variances += [normal_var]
            '''
            hit = np.array(hit)
            ray = hit - origin            
            ray_range = np.linalg.norm(ray)        
            range_inv = 1.0 / ray_range
            t_truncation_distance = tsdf.truncation_distance * range_inv
            t_start = 1.0 - t_truncation_distance
            t_end = 1.0 + t_truncation_distance
            grid_index, grid_step, t_max, t_delta = getRaytracingHelperVariables(origin, ray, t_start,t_end, 1. / tsdf.resolution)
            t = 0
            while t < 1.0 :                
                #print('t',t,'t_max',t_max,'t_delta',t_delta)     
                #print('grid_index',grid_index)
                t_next = np.min(t_max)
                min_coeff_idx = np.argmin(t_max)
                sampling_point = grid_index * tsdf.resolution #origin + (t + t_next)/2 * ray
                #print('sampling_point',sampling_point,'t',origin + (t) * ray,'tn',origin + (t_next) * ray)
                cell_index = tsdf.getCellIndexAtPosition(sampling_point)
                cell_center = tsdf.getPositionAtCellIndex(cell_index)
                distance = np.linalg.norm(cell_center - origin)
                self.updateCell(tsdf, cell_index, ray_range - distance, ray_range, 1)                
                #print('cell_index', cell_index)      
                t = t_next
                grid_index[min_coeff_idx] += grid_step[min_coeff_idx]
                t_max[min_coeff_idx] += t_delta[min_coeff_idx]
        #self.drawScanWithNormals(hits, normal_orientations, origin, normal_estimation_weight_sums)
