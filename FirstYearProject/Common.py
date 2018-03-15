import numpy
import math
import os

class consts:
    c = 1 #speed of light
    h = 2*math.pi # plancs constant
    Na = 6.022*10**23 #advokadokonstanten
    me = 0.510 #[MeV/c^2] Mass of electron
    e = 1 #electron charge

class particleData:
    def __init__(self, mass, label, halflife):
        self.m = mass
        self.Label = label
        self.halflife = halflife

def velocity(P,m):
    return P/numpy.sqrt(m**2+P**2) #[c] derived from P=gamma*m*v=1/sqrt(1-v^2/c^2)*m*v, c=1


def list_files(dir):
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            r.append(os.path.join(root, name))
    return r  

particles = [particleData(0.51, "Electron", math.inf),
             particleData(106, "Muon", 2.196*10**-6*math.log(2)),
             particleData(139.6, "Pion", 2.6*10**-6*math.log(2)),
             particleData(493.7, "Kaon", 1.24*10**-8*math.log(2)), 
             particleData(938, "Proton", math.inf), 
             particleData(1876, "Deuterium", math.inf),
             particleData(2814, "Tritium", math.inf),]