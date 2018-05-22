import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib import pyplot
from shapely.geometry import LineString

class Rangefinder:    
    def __init__(self):
        self.angular_resultion = 0.08
        self.min_angle = -1.5
        self.max_angle = 1.5
        self.max_range = 10
        self.range_variance = 0.03
        self.angular_variance = 0.0
        np.random.seed(42)
        
    def scan(self, environment, origin):
        hits = []
        for ray_angle in np.arange(self.min_angle, self.max_angle, self.angular_resultion):
            ray_angle_noise = np.random.normal(0, self.angular_variance, 1)
            ray_angle += ray_angle_noise
            dx = np.sin(ray_angle)
            dy = np.cos(ray_angle)        
            ray = (origin[0] + self.max_range * dx, origin[1] + self.max_range * dy)
            ray_line = LineString([origin, ray])        
            #x, y = ray_line.xy
            #ax.plot(x, y, color='black',  linewidth=1)
            intersect = ray_line.intersection(environment)
            if not intersect.is_empty:        
                ray_length = np.linalg.norm(intersect.coords[0])
                range_noise = np.random.normal(0, self.range_variance, 1)                
                hits += [intersect.coords[0]*((ray_length+range_noise)/ray_length)]
        return hits
