import bresenham
from math import sin, cos, pi,tan, atan2,log
import math
from itertools import groupby
from operator import itemgetter
import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import MapMetaData
from geometry_msgs.msg import Pose
import math
from sensor_msgs.msg import PointCloud2, PointField
import struct
import ctypes
import time

_DATATYPES = {}
_DATATYPES[PointField.INT8]    = ('b', 1)
_DATATYPES[PointField.UINT8]   = ('B', 1)
_DATATYPES[PointField.INT16]   = ('h', 2)
_DATATYPES[PointField.UINT16]  = ('H', 2)
_DATATYPES[PointField.INT32]   = ('i', 4)
_DATATYPES[PointField.UINT32]  = ('I', 4)
_DATATYPES[PointField.FLOAT32] = ('f', 4)
_DATATYPES[PointField.FLOAT64] = ('d', 8)

def _get_struct_fmt(is_bigendian, fields, field_names=None):
     fmt = '>' if is_bigendian else '<'
 
     offset = 0
     for field in (f for f in sorted(fields, key=lambda f: f.offset) if field_names is None or f.name in field_names):
         if offset < field.offset:
             fmt += 'x' * (field.offset - offset)
             offset = field.offset
         if field.datatype not in _DATATYPES:
             print('Skipping unknown PointField datatype [%d]' % field.datatype, file=sys.stderr)
         else:
             datatype_fmt, datatype_length = _DATATYPES[field.datatype]
             fmt    += field.count * datatype_fmt
             offset += field.count * datatype_length
 
     return fmt

def read_points(cloud, field_names=None, skip_nans=False, uvs=[]):
     """
     Read points from a L{sensor_msgs.PointCloud2} message.
 
00064     @param cloud: The point cloud to read from.
00065     @type  cloud: L{sensor_msgs.PointCloud2}
00066     @param field_names: The names of fields to read. If None, read all fields. [default: None]
00067     @type  field_names: iterable
00068     @param skip_nans: If True, then don't return any point with a NaN value.
00069     @type  skip_nans: bool [default: False]
00070     @param uvs: If specified, then only return the points at the given coordinates. [default: empty list]
00071     @type  uvs: iterable
00072     @return: Generator which yields a list of values for each point.
00073     @rtype:  generator
     """
     fmt = _get_struct_fmt(cloud.is_bigendian, cloud.fields, field_names)
     width, height, point_step, row_step, data, isnan = cloud.width, cloud.height, cloud.point_step, cloud.row_step, cloud.data, math.isnan
     unpack_from = struct.Struct(fmt).unpack_from
     if skip_nans:
         if uvs:
             for u, v in uvs:
                 p = unpack_from(data, (row_step * v) + (point_step * u))
                 has_nan = False
                 for pv in p:
                     if isnan(pv):
                         has_nan = True
                         break
                 if not has_nan:
                     yield p
         else:
             for v in range(height):
                 offset = row_step * v
                 for u in range(width):
                     p = unpack_from(data, offset)
                     has_nan = False
                     for pv in p:
                         if isnan(pv):
                             has_nan = True
                             break
                     if not has_nan:
                         yield p
                     offset += point_step
     else:
         if uvs:
             for u, v in uvs:
                 yield unpack_from(data, (row_step * v) + (point_step * u))
         else:
             for v in range(height):
                 offset = row_step * v
                 for u in range(width):
                     yield unpack_from(data, offset)
                     offset += point_step

class Localmap:
    def __init__(self, height = 100, width = 100, resolution= 0.05):
        self.height=height
        self.width=width
        self.resolution=resolution
        self.punknown=-1
        self.morigin = (width//2, height//2)
        self.localmap=[self.punknown]*int(self.width/self.resolution)*int(self.height/self.resolution)
        self.logodds=[0.0]*int(self.width/self.resolution)*int(self.height/self.resolution)
        self.origin=int(math.ceil(self.morigin[0]/resolution))+int(math.ceil(width/resolution)*math.ceil(self.morigin[1]/resolution))
        self.max_scan_range=1.0
        self.map_origin= (math.floor(self.width/(2*self.resolution)), math.floor(self.height/(2*self.resolution)))

    def resize(self, scale = 4):
        old_height = self.height
        old_width = self.width
        self.height *= scale
        self.width *= scale
        prev_map_origin = self.map_origin
        self.map_origin = (math.floor(self.width/(2*self.resolution)), math.floor(self.height/(2*self.resolution)))
        prev_origin = self.origin
        self.origin = int(self.map_origin[0]) + int(math.floor(self.map_origin[1]*self.width))
        new_local_map = [self.punknown]*int(self.width/self.resolution)*int(self.height/self.resolution)
        for i in range(len(self.localmap)):
            col_x = int(i % (old_width/self.resolution)) - prev_map_origin[0]
            row_y = int(i // (old_width/self.resolution)) - prev_map_origin[1]
            val = self.localmap[i]
            new_ind = col_x + self.map_origin[0] + int((row_y + self.map_origin[1])*self.width/self.resolution)
            new_local_map[new_ind] = val





    def updatemap(self, pcl, range_max):

        #robot_origin=int(pose[0])+int(math.ceil(self.width/self.resolution)*pose[1])
        cloud_it = list(read_points(pcl, field_names = ('x', 'y', 'z'), skip_nans = True))

        for p in cloud_it:
            px = int(p[0]/self.resolution)
            py = int(p[2]/self.resolution)

            l = list(bresenham.bresenham(0,0,px,py))
            for j in range(len(l)):                    
                lpx= int(self.morigin[0]/self.resolution + l[j][0])
                lpy=int(self.morigin[1]/self.resolution + l[j][1])

                if (0<=lpx<self.width/self.resolution and 0<=lpy<self.height/self.resolution):
                    index= int(lpx + math.floor(self.width/self.resolution)*(lpy))
                    rad = math.sqrt(p[0]**2 + p[2]**2)
                    if(j<len(l)-1):
                        if self.localmap[index] == -1:
                            self.localmap[index] = 0
                    else:
                        if rad<self.max_scan_range*range_max:
                            self.localmap[index] = 100





class GridMapPubSub(Node):
    def __init__(self):
        super().__init__('gridmap')
        self.publisher_ = self.create_publisher(OccupancyGrid, '/gridmap', 1)
        self.subscription = self.create_subscription(
            PointCloud2,
            '/modpointcloud',
            self.sub_callback,
            5)
        self.lmap = Localmap()
        self.og = OccupancyGrid()
        self.og.data = self.lmap.localmap
        self.og.info = MapMetaData()
        self.og.info.resolution = self.lmap.resolution
        self.og.info.width = int(self.lmap.width/self.lmap.resolution)
        self.og.info.height = int(self.lmap.height/self.lmap.resolution)
        self.og.info.origin = Pose()
        self.og.info.origin.position.x = float(self.lmap.morigin[0])
        self.og.info.origin.position.y = float(self.lmap.morigin[1])
        self.subscription  # prevent unused variable warning


    def sub_callback(self, pcl):
        self.curr_data = pcl
        self.lmap.updatemap(pcl, 6)
        self.og.data = self.lmap.localmap
        self.publisher_.publish(self.og)




def main():
    rclpy.init()
    grid_map_node = GridMapPubSub()
    
    
    rclpy.spin(grid_map_node)
    grid_map_node.destroy_node()
    rclpy.shutdown()
    



if __name__ == "__main__":
    main()

