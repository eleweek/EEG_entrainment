import sys
from pprint import pprint

import pyxdf

filepath = sys.argv[1]

streams, _ = pyxdf.load_xdf(filepath)


for stream in streams:
    pprint(stream['info'])