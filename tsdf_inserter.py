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

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def toTwoPi(value):
    if(value > math.pi):
        return toTwoPi(value - 2 * math.pi) #value % math.pi
    if(value < - math.pi):
        return toTwoPi(value + 2 * math.pi) #value % -math.pi
    return value

def angle_between(v1, v2):
    v1_normalized = unit_vector(v1)
    v2_normalized = unit_vector(v2)
    return toTwoPi(angle(v1)-angle(v2))

def angle(v):
    v_normalized = unit_vector(v)
    return math.atan2(v_normalized[1], v_normalized[0])    
  
def distanceLinePoint(line_p0, line_p1, point):
    numerator = np.abs((line_p1[1]-line_p0[1])*point[0] - (line_p1[0]-line_p0[0])*point[1] + line_p1[0]*line_p0[1] - line_p1[1]*line_p0[0]) 
    denominator = np.linalg.norm(line_p1-line_p0)
    return numerator/denominator  

def gaussian(x, mu=0, sigma=1):
    return 1/(math.sqrt(2*math.pi)*sigma**2)*math.e**(-0.5*((x-mu)/sigma**2)**2)

class ScanNormalTSDFRangeInserter:   
    
    def __init__(self, use_normals_weight=False, n_normal_samples=8, default_weight=1, use_distance_cell_to_observation_weight=False, use_distance_cell_to_ray_weight=False, use_scale_distance=False, normal_distance_factor=1, max_weight=1000, draw_normals_scan_indices=[0]):
        self.use_normals_weight = use_normals_weight
        self.use_distance_cell_to_observation_weight = use_distance_cell_to_observation_weight
        self.sigma_distance_cell_to_observation_weight = 0.7
        self.use_distance_cell_to_ray_weight = use_distance_cell_to_ray_weight
        self.sigma_distance_cell_to_ray_weight = 0.8
        self.n_normal_samples = n_normal_samples
        self.default_weight = default_weight
        self.normal_distance_factor = normal_distance_factor #0 --> all normals same weight, 1 --> f(0)=1, f(0.1)=0.9 f(0.2)=0.82 independent of distance, inf -->only closest normal counts
        self.max_weight = max_weight
        self.draw_normals_scan_indices = draw_normals_scan_indices
        self.num_inserted_scans = 0
        self.use_scale_distance = use_scale_distance
        print(self)
        
    def __str__(self):
        return "ScanNormalTSDFRangeInserter \n use_normals_weight %s \n n_normal_samples %s\n default_weight %s\n normal_distance_factor %s\n" % (self.use_normals_weight, self.n_normal_samples, self.default_weight, self.normal_distance_factor)
        
        
    def updateCell(self, tsdf, cell_index, update_distance, ray_length, update_weight):       
        if(abs(update_distance)< tsdf.truncation_distance):
            updated_weight = min(tsdf.getWeight(cell_index) + update_weight, self.max_weight)
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
            delta_sample = sample-neighbor
            if(delta_sample[0] < 0):
                delta_sample = -delta_sample
            
            tangent_angle = angle(delta_sample)
            if(np.abs(tangent_angle) > math.pi/2):
                print('tangent out of interval', tangent_angle)
            #print('tangent', tangent_angle)
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
        
    def drawScanWithNormals(self, hits, normal_orientations, sensor_origin, normal_weights, normal_variances, normal_angle_to_ray):
        fig = plt.figure()
        ax = plt.subplot(411)
        x_val = [x[0] for x in hits]
        y_val = [x[1] for x in hits]
        sc = ax.scatter(x_val, y_val, c=normal_weights, marker='x', cmap=cm.jet)
        plt.colorbar(sc)
        ax.scatter(sensor_origin[0], sensor_origin[1], marker='x')
        for idx, normal_orientation in enumerate(normal_orientations):  
            normal_scale = 0.1
            dx = normal_scale*np.cos(normal_orientation)
            dy = normal_scale*np.sin(normal_orientation)        
            ax.arrow(x_val[idx], y_val[idx], dx, dy, fc='k', ec='k', color='b')
            ax.set_aspect('equal')        
        plt.title('Normal Estimation Weights')
            
        ax = plt.subplot(412)
        x_val = [x[0] for x in hits]
        y_val = [x[1] for x in hits]
        sc = ax.scatter(x_val, y_val, c=normal_variances, marker='x', cmap=cm.jet)
        plt.colorbar(sc)
        ax.scatter(sensor_origin[0], sensor_origin[1], marker='x')
        for idx, normal_orientation in enumerate(normal_orientations):  
            normal_scale = 0.1
            dx = normal_scale*np.cos(normal_orientation)
            dy = normal_scale*np.sin(normal_orientation)        
            ax.arrow(x_val[idx], y_val[idx], dx, dy, fc='k', ec='k', color='b')
            ax.set_aspect('equal')
        plt.title('Normal Estimation Variances')
            
        ax = plt.subplot(413)
        x_val = [x[0] for x in hits]
        y_val = [x[1] for x in hits]
        sc = ax.scatter(x_val, y_val, c=np.cos(normal_angle_to_ray), marker='x', cmap=cm.jet)
        plt.colorbar(sc)
        ax.scatter(sensor_origin[0], sensor_origin[1], marker='x')
        for idx, normal_orientation in enumerate(normal_orientations):  
            normal_scale = 0.1
            dx = normal_scale*np.cos(normal_orientation)
            dy = normal_scale*np.sin(normal_orientation)        
            ax.arrow(x_val[idx], y_val[idx], dx, dy, fc='k', ec='k', color='b')
            ax.set_aspect('equal')
        plt.title('Angle normal to ray')
            
        ax = plt.subplot(414)
        x_val = [x[0] for x in hits]
        y_val = [x[1] for x in hits]
        combined_weights = np.reciprocal(np.sqrt(np.array(normal_variances))) * (np.square(np.array(normal_weights))) * np.square(np.cos(normal_angle_to_ray))
        combined_weights = np.cos(normal_angle_to_ray)
        sc = ax.scatter(x_val, y_val, c=combined_weights, marker='x', cmap=cm.jet)
        plt.colorbar(sc)
        ax.scatter(sensor_origin[0], sensor_origin[1], marker='x')
        for idx, normal_orientation in enumerate(normal_orientations):  
            normal_scale = 0.1
            dx = normal_scale*np.cos(normal_orientation)
            dy = normal_scale*np.sin(normal_orientation)        
            ax.arrow(x_val[idx], y_val[idx], dx, dy, fc='k', ec='k', color='b')
            ax.set_aspect('equal')
        plt.title('Combined weight')
        
    
    def insertScan(self, tsdf, hits, origin):
        origin = np.array(origin)
        hits = np.array(hits)
        n_hits = len(hits)
        normal_orientations = []
        normal_orientation_variances = []
        normal_estimation_weight_sums = []
        normal_estimation_angles_to_ray = []
        normal_estimation_angle_to_ray = 0
        normal_orientation = 0
        for idx, hit in enumerate(hits):      
            #print('origin',origin)       
            #print('hit',hit)      
            hit = np.array(hit)
            ray = hit - origin    
            
            if self.use_normals_weight or True:
                neighbor_indices = np.array(list(range(idx-int(np.floor(self.n_normal_samples/2)), idx)) + list(range(idx+1, idx+int(np.ceil(self.n_normal_samples/2) + 1))))
                neighbor_indices = neighbor_indices[neighbor_indices >= 0]
                neighbor_indices = neighbor_indices[neighbor_indices < n_hits]
                normal_orientation, normal_var, normal_estimation_weight_sum = self.computeNormal(hit, hits[neighbor_indices])
                normal_orientations += [normal_orientation]
                normal_estimation_weight_sums += [normal_estimation_weight_sum]
                normal_orientation_variances += [normal_var]
                normal_estimation_angle_to_ray = normal_orientation - angle(-ray)
                normal_estimation_angles_to_ray += [normal_estimation_angle_to_ray] # 
             
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
                distance_cell_center_to_origin = np.linalg.norm(cell_center - origin)
                update_weight = 1
                update_distance = ray_range - distance_cell_center_to_origin
                #use_distance_cell_to_observation_weight
                if self.use_normals_weight:
                    update_weight = np.cos(normal_estimation_angle_to_ray)
                if self.use_distance_cell_to_observation_weight:
                    normalized_distance_cell_to_observation = np.abs(ray_range - distance_cell_center_to_origin)/tsdf.resolution
                    distance_cell_to_observation_weight = gaussian(normalized_distance_cell_to_observation, 0, self.sigma_distance_cell_to_observation_weight)                    
                    distance_cell_to_observation_weight = (tsdf.truncation_distance - np.abs(ray_range - distance_cell_center_to_origin))/tsdf.truncation_distance
                    update_weight *= distance_cell_to_observation_weight
                if self.use_distance_cell_to_ray_weight:
                    distance_cell_to_ray = distanceLinePoint(origin, hit, cell_center)/tsdf.resolution
                    #distance_cell_to_ray_weight = distance_cell_to_ray
                    distance_cell_to_ray_weight = gaussian(distance_cell_to_ray, 0, self.sigma_distance_cell_to_ray_weight)                    
                    update_weight *= distance_cell_to_ray_weight
                
                if self.use_scale_distance:
                    #print(np.array([np.cos(normal_orientation), np.sin(normal_orientation)]))
                    update_distance = (cell_center - hit).dot(np.array([np.cos(normal_orientation), np.sin(normal_orientation)]))
                    
                
                self.updateCell(tsdf, cell_index, update_distance , ray_range, update_weight)
                #print('cell_index', cell_index)      
                t = t_next
                grid_index[min_coeff_idx] += grid_step[min_coeff_idx]
                t_max[min_coeff_idx] += t_delta[min_coeff_idx]
        if self.use_normals_weight:
            if  self.num_inserted_scans in self.draw_normals_scan_indices :
                self.drawScanWithNormals(hits, normal_orientations, origin, normal_estimation_weight_sums, normal_orientation_variances, normal_estimation_angles_to_ray)
                self.draw_normals = False
                #print('avg normal error', np.mean(np.abs(normal_orientations)))
                pass
        self.num_inserted_scans += 1
