"""
https://www.osgeo.cn/shapely/manual.html
"""
from shapely.geometry import Polygon, LinearRing

def polygon_extend(coords, radius):
    r = LinearRing(coords)
    s = Polygon(r)
    t = Polygon(s.buffer(radius).exterior, [r])
    return list(t.exterior.coords)
