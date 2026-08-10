import numpy as np
from numba import jit, prange
from scipy.optimize import minimize
from scipy.interpolate import splev, splprep
from typing import List



class StreamlineIntersectionSplitter:
    def __init__(self, offset_boundingBox=0.05, num_samples=5, tolerance=1e-8):
        self.offset_boundingBox = offset_boundingBox
        self.num_samples = num_samples
        self.tolerance = tolerance
    
    def process_streamlines(self, streamlines: List):
        """Main method to find intersections and split streamlines."""

        splines         = self.get_streamlines_as_splines(streamlines)
        split_points    = self.find_all_intersections(splines)
        new_splines     = self.split_streamlines(splines, split_points)
        return self.splines_to_points(new_splines)
        
    def find_all_intersections(self, splines):
        """Find all intersection points between streamlines."""
        split_points = {i: [] for i in range(len(splines))}
        
        for i in range(len(splines)):
            for j in range(i + 1, len(splines)):
                if self.check_endpoints_match(splines[i], splines[j]):
                    continue
                
                intersection = self.find_intersection(splines[i], splines[j])
                if intersection:
                    t1, t2 = intersection
                    split_points[i].append(t1)
                    split_points[j].append(t2)
        
        return split_points
    
    def find_intersection(self, spline1, spline2):
        """Find intersection between two splines."""
        tck1, u1 = spline1
        tck2, u2 = spline2
        
        u1_fine = np.linspace(0, 1, self.num_samples)
        u2_fine = np.linspace(0, 1, self.num_samples)
        points1 = np.array(splev(u1_fine, tck1)).T
        points2 = np.array(splev(u2_fine, tck2)).T
        
        potential = self.check_bounding_boxes(points1, points2, u1_fine, u2_fine)
        if len(potential) > 0:
            return self.confirm_intersection(potential, tck1, tck2)
        return None
    
    def check_endpoints_match(self, spline1, spline2):
        """Check if splines share endpoints."""
        tck1, _ = spline1
        tck2, _ = spline2
        
        p1_start = np.array(splev(0, tck1))
        p1_end = np.array(splev(1, tck1))
        p2_start = np.array(splev(0, tck2))
        p2_end = np.array(splev(1, tck2))
        
        return (np.allclose(p1_start, p2_start, atol=self.tolerance) or 
                np.allclose(p1_start, p2_end, atol=self.tolerance) or
                np.allclose(p1_end, p2_start, atol=self.tolerance) or 
                np.allclose(p1_end, p2_end, atol=self.tolerance))
    
    def confirm_intersection(self, potential_intersections, tck1, tck2):
        """Optimize to confirm actual intersection."""
        for u1_init, u2_init in potential_intersections:
            dist = lambda t: np.sum((np.array(splev(t[0], tck1)) - 
                                    np.array(splev(t[1], tck2)))**2)
            res = minimize(dist, [u1_init, u2_init], 
                         bounds=[(0,1), (0,1)], method='L-BFGS-B')
            if res.fun < self.tolerance:
                return (res.x[0], res.x[1])
        return None
        
    def splines_to_points(self, splines, num_points=100):
        """Convert splines back to point arrays."""
        result = []
        for spline in splines:
            tck, u = spline  # Now this works with lists [tck, u]
            u_eval = np.linspace(0, 1, num_points)
            points = np.array(splev(u_eval, tck)).T
            result.append(points)
        return result

    def split_streamlines(self, splines, split_points):
        """Split streamlines at intersection points."""
        result = []
        
        for i, spline in enumerate(splines):
            if not split_points[i]:
                result.append(spline)  # This is [tck, u] list
            else:
                splits = sorted(split_points[i])
                tck, u = spline
                
                splits = [0] + [t for t in splits if 0.001 < t < 0.999] + [1]
                
                for k in range(len(splits) - 1):
                    t_start = splits[k]
                    t_end = splits[k + 1]
                    
                    t_sub = np.linspace(t_start, t_end, 20)
                    points_sub = np.array(splev(t_sub, tck))
                    
                    if points_sub.shape[1] > 3:
                        tck_new, u_new = splprep(points_sub, s=0)
                        result.append([tck_new, u_new]) 
    
        return result
            
    def check_bounding_boxes(self, points1, points2, u1_fine, u2_fine):
        """Wrapper for parallel bounding box check."""
        return check_bounding_boxes_parallel(points1, points2, u1_fine, u2_fine, 
                                             self.offset_boundingBox)
    
    def get_streamlines_as_splines(self, streamlines):

        splines = []

        for i in range(len(streamlines)):
            streamline = np.array(streamlines[i])
            x = streamline[:, 0]
            y = streamline[:, 1]
            
            m = x.shape[0]

            k = min(3,m-1)
            
            if m<2:
                continue

            tck, u = splprep([x, y], s=0, k=k)  # k=1 linear splines
            splines.append([tck, u])
            
        return splines

# Standalone parallel function (can't be a class method with numba)
@jit(nopython=True, parallel=True)
def check_bounding_boxes_parallel(points1, points2, u1_fine, u2_fine, offset_boundingBox):
    n1 = len(points1) - 1
    n2 = len(points2) - 1
    results = np.zeros((n1 * n2, 2))
    
    for idx in prange(n1 * n2):
        i = idx // n2
        j = idx % n2
        
        min_x1, max_x1 = min(points1[i][0], points1[i+1][0]), max(points1[i][0], points1[i+1][0])
        min_y1, max_y1 = min(points1[i][1], points1[i+1][1]), max(points1[i][1], points1[i+1][1])
        min_x2, max_x2 = min(points2[j][0], points2[j+1][0]), max(points2[j][0], points2[j+1][0])
        min_y2, max_y2 = min(points2[j][1], points2[j+1][1]), max(points2[j][1], points2[j+1][1])
        
        min_x11 = min_x1 - max(offset_boundingBox, offset_boundingBox * abs(min_x1))
        max_x11 = max_x1 + max(offset_boundingBox, offset_boundingBox * abs(max_x1))
        min_y11 = min_y1 - max(offset_boundingBox, offset_boundingBox * abs(min_y1))
        max_y11 = max_y1 + max(offset_boundingBox, offset_boundingBox * abs(max_y1))
        min_x22 = min_x2 - max(offset_boundingBox, offset_boundingBox * abs(min_x2))
        max_x22 = max_x2 + max(offset_boundingBox, offset_boundingBox * abs(max_x2))
        min_y22 = min_y2 - max(offset_boundingBox, offset_boundingBox * abs(min_y2))
        max_y22 = max_y2 + max(offset_boundingBox, offset_boundingBox * abs(max_y2))
        
        if (min_x11 <= max_x22 and max_x11 >= min_x22 and
            min_y11 <= max_y22 and max_y11 >= min_y22):
            results[idx] = [u1_fine[i], u2_fine[j]]
        else:
            results[idx] = [-1, -1]
    
    return results[results[:, 0] >= 0]

