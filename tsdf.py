import numpy as np

class TSDF:
    def __init__(self):
        self.resolution = 0.1
        self.size = 10.0
        self.truncation_distance = 0.3
        #we assume a grid with the origin at (0,0) only defined in positive direction
        n_cells_per_dimension = int(self.size/self.resolution + 1)
        self.tsdf = np.zeros((n_cells_per_dimension, n_cells_per_dimension))
        self.weights = np.zeros((n_cells_per_dimension, n_cells_per_dimension))
    
    def getCellIndexAtPosition(self, position):
        cell_index =  (np.round(np.array(position)/self.resolution)).astype(int)  
        return [[cell_index[1]],[cell_index[0]]]
    
    def getPositionAtCellIndex(self, cell_index):        
        return np.array((cell_index[1][0], cell_index[0][0])) * self.resolution
    
    def setTSDF(self, position_index, value):
        self.tsdf[position_index] = value;
        
    def setWeight(self, position_index, value):
        self.weights[position_index] = value;
