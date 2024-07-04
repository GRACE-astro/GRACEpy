import numpy as np

class line:
    def __init__(self,point,direction):
        """
        Initialize the line with a point and a direction vector.
        
        :param point: A 3-element array-like representing a point on the line.
        :param direction: A 3-element array-like representing the direction vector of the line.
        """
        point = np.pad(np.array(point),(0,3-len(np.array(point))),constant_values=(0))
        direction = np.pad(np.array(direction),(0,3-len(np.array(direction))),constant_values=(0))
        self.__point = point 
        self.__direction = direction 
    def point(self):
        return self.__point 
    def direction(self):
        return self.__direction
    def intersect_with_plane(self, plane):
        """
        Find the intersection point of the line with a plane.
        
        :param plane: A Plane object.
        :return: A 3-element numpy array representing the intersection point, or None if no intersection.
        """
        denom = np.dot(plane.normal, self.direction)
        
        if np.abs(denom) < 1e-6:
            # Line is parallel to the plane
            return None
        
        t = -(np.dot(plane.normal, self.point) + plane.D) / denom
        intersection_point = self.point + t * self.direction
        return intersection_point
    
    def is_point_on_line(self, point):
        """
        Check if a given point lies on the line.
        
        :param point: A 3-element array-like representing the point.
        :return: True if the point lies on the line, False otherwise.
        """
        point = np.array(point)
        if np.all(self.direction == 0):
            return False
        
        t_values = (point - self.point) / self.direction
        # Check if t_values are all the same, considering floating point precision
        return np.allclose(t_values, t_values[0], atol=1e-6)

import numpy as np

class plane:
    def __init__(self, normal, point):
        """
        Initialize the plane with a normal vector and a point on the plane.
        
        :param normal: A 3-element array-like representing the normal vector of the plane.
        :param point: A 3-element array-like representing a point on the plane.
        """
        point = np.pad(np.array(point),(0,3-len(np.array(point))),constant_values=(0))
        normal = np.pad(np.array(normal),(0,3-len(np.array(normal))),constant_values=(0))
        self.normal = np.array(normal)
        self.point = np.array(point)
    
    def equation(self):
        """
        Returns the plane equation coefficients in the form:
        (x - x0)n1 + (y - y0)n2 + (z - z0)n3 = 0
        
        :return: A tuple (normal, point) where normal is the normal vector (n1, n2, n3)
                 and point is a point on the plane (x0, y0, z0).
        """
        return (self.normal, self.point)
    
    def intersect_with_line(self, line):
        """
        Find the intersection point of the plane with a line.
        
        :param line_point: A 3-element array-like representing a point on the line.
        :param line_direction: A 3-element array-like representing the direction vector of the line.
        :return: A 3-element numpy array representing the intersection point, or None if no intersection.
        """
        line_point = line.point()
        line_direction = line.direction()
        
        denom = np.dot(self.normal, line_direction)
        
        if np.abs(denom) < 1e-6:
            # Line is parallel to the plane
            return None
        
        t = np.dot(self.normal, self.point - line_point) / denom
        intersection_point = line_point + t * line_direction
        return intersection_point
    
    def intersect_with_plane(self, other):
        """
        Find the line of intersection between this plane and another plane.
        
        :param other: Another Plane object.
        :return: A tuple (point, direction) representing the point and direction of the intersection line.
        """
        direction = np.cross(self.normal, other.normal)
        
        if np.linalg.norm(direction) < 1e-6:
            # Planes are parallel or coincident
            return None
        
        # Find a point on the line of intersection
        A = np.array([self.normal, other.normal, direction])
        d = np.array([np.dot(self.normal, self.point), np.dot(other.normal, other.point), 0])
        
        if np.linalg.matrix_rank(A) < 3:
            # Planes are parallel or coincident
            return None
        
        point = np.linalg.solve(A, d)
        return (point, direction)
    
    def is_point_on_plane(self, point):
        """
        Check if a given point lies on the plane.
        
        :param point: A 3-element array-like representing the point.
        :return: True if the point lies on the plane, False otherwise.
        """
        return np.abs(np.dot(self.normal, point - self.point)) < 1e-6
    
    def is_line_in_plane(self,line):
        """
        Determine if a line lies within this plane.
        """
        line_point = line.point()
        line_direction = line.direction()
        if not self.is_point_on_plane(line_point):
            return False
        if abs(np.dot(line_direction,self.normal)) > 1e-06:
            return False 
        return True



def find_intersecting_plane(_plane,_line):
    """
    Find a plane whose intersection with another is the given line.
    :param plane: The plane to intersect with.
    :param line: The line which needs to be the intersection.
    :return: A plane whose intersection with the first one yields the line.
    """
    if not _plane.is_line_in_plane(_line):
        raise ValueError("The line is not contained within the plane.")
    (normal,_) = _plane.equation()
    other_normal = np.cross(_line.direction(),normal)
    
    return plane(other_normal, _line.point())


def find_normal_vector(v):
    # Given vector v
    v = pad_array_with_zeros(v)
    
    # Choose an arbitrary vector that is not collinear with v
    if np.allclose(v, [1, 0, 0]):
        arbitrary_vector = np.array([0, 1, 0])
    else:
        arbitrary_vector = np.array([1, 0, 0])
    
    # Compute the cross product
    normal_vector = np.cross(v, arbitrary_vector)
    
    # Check if the resulting vector is zero, if so, choose a different arbitrary vector
    if np.allclose(normal_vector, [0, 0, 0]):
        arbitrary_vector = np.array([0, 1, 0])
        normal_vector = np.cross(v, arbitrary_vector)
    
    return normal_vector

def pad_array_with_zeros(a,target_size=3):
    return np.pad(np.array(a),(0,target_size-len(np.array(a))),constant_values=(0))

    