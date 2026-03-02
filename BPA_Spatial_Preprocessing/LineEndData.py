from enum import Enum
from shapely.geometry import Point, LineString

DEBUG = False

class ConnectType(Enum):
    FREE = 0
    LINE = 1
    SUBSTATION = 2
    SELF = 3

class LineEnd(Enum):
    START = 0
    END = 1

class LineEndData:
    def __init__(self, idx, end_idx, associated_line_str, num_pts=2):
        self.free_point = True
        self.connection_type = ConnectType.FREE
        self.substation_name = None
        #self.merging_line_xrefcd = None
        self.all_merging_lines_xrefcd = set()
        if not isinstance(associated_line_str, LineString):
            raise ValueError("associated_line_str must be a LineString")
        # If end_idx is 0 take first two points, else take last two points
        if end_idx == LineEnd.START:
            self.point = Point(associated_line_str.coords[0])
            self.associated_line_str = LineString(associated_line_str.coords[:num_pts])
        else:
            self.point = Point(associated_line_str.coords[-1])
            self.associated_line_str = LineString(associated_line_str.coords[-num_pts:])
        self.line_idx = idx
        self.end = end_idx
    def __str__(self):
        return f'[\nPoint: {self.point}\nFree: {self.free_point}\nConnectionType: {self.connection_type}\nSubstation: {self.substation_name}\nMergingLines: {self.all_merging_lines_xrefcd}\n]'
    __repr__ = __str__

