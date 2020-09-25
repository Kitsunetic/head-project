import math
from dataclasses import dataclass


@dataclass
class History:
    loss = []
    yaw = []
    pitch = []
    roll = []
    rms = []
    diff = []


def radian2degree(radian):
    return radian * 180 / math.pi
