from math import sqrt
from pprint import pprint
import numpy as np
from networktables import NetworkTables

NetworkTables.startClient("10.212.118.22")
NetworkTables.initialize()

sd = NetworkTables.getTable("chameleon-vision").getSubTable("USB Camera-B4.09.24.1")

# while True:
#     val = sd.getValue("aux_targets", "None")
#     if val == "None":
#         continue
#     targets = [""]
#     count = 0
#     for letter in val[1:-1]:
#         targets[-1] += letter
#         if letter == "]":
#             targets.append("")
#     vectors = []
#     for target in targets:
#         target = target[2:-1]
#         if "{" in target:
#             index = target.index("{")
#             rotation = target[target[index:].index("rotation"):]
#             translation = target[index:target[index:].index("rotation")]
#             print(rotation + "r")
#             print(translation)
#
# # print(sd.getValue("", "foo"))
while True:
    gotten = str(sd.getValue("targetPose", "None"))
    if gotten == "None":
        continue
    gotten = gotten[1:-1]
    gotten = gotten.split(",")
    x, y, a = list(map(lambda x: float(x[:-1]), gotten))
    print(sqrt((x*x) + (y*y)))
