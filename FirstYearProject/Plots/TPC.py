import math
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import BetheBlock
import Common
import uproot
from multiprocessing import Pool

def get_file_histogram(file):
    tracks = uproot.open(file)["filterTracks"]["fTreeOutput"]
    H, xedges, yedges = np.histogram2d(tracks["fdMomentumTPC"].array()*1000/tracks["fiCharge"].array(),
           tracks["fdTPCsignal"].array()/17000, 
           bins=(500,500), 
           range=((-1500, 1500), (0, 0.1)))
    print ("done processing " + file)
    return H

def main():
    p = Pool(8)
    histogram = sum(p.map(get_file_histogram,Common.list_files("Data")))

    fig, ax = plt.subplots()
    plt.imshow(histogram.transpose(), interpolation='nearest', origin='low', norm=colors.LogNorm(), extent= (-1500, 1500, 0, 0.1), aspect="auto")
    plt.colorbar()

    #draw expected values of de/dx of a tpc momentum per particle hypopthesis
    xMomentum  = np.linspace(-1500,1500, num=1000) #make a list of numbers representing momentum in MeV to use for graphing

    for (particle,fit) in zip(Common.particles,BetheBlock.fitParameters):
        calculatedPlot = BetheBlock.multipleGasBethe(xMomentum, particle.m, [BetheBlock.Argon, BetheBlock.CO2], [0.88,0.12])
        axis = ax.plot(xMomentum, calculatedPlot, label=particle.Label, linewidth=2)

    ax.set_title('Argon gas, # of tracks: '+ str(histogram.sum()))
    ax.set_xlabel('P/z (Rigidity) [MeV/c]')
    ax.set_ylabel('-dE/dv [MeV/cm]')
    ax.set_ylim([0,0.1])
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()