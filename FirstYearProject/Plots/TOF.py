import math
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import uproot
import Common
from multiprocessing import Pool

def get_file_histogram(file):
    tracks = uproot.open(file)["filterTracks"]["fTreeOutput"]
    # make a filter to avoid bad tracks
    TOFSignal = tracks["fdTOFsignal"].array()
    momentumTPC = tracks["fdMomentumTPC"].array()*1000
    psudorapidity = tracks["fdEta"].array()
    filter = np.logical_and.reduce((tracks["fbTPC"].array(),tracks["fbTOF"].array(),TOFSignal < 40000, TOFSignal > 11000, momentumTPC/np.cosh(psudorapidity) > 550))
    TOF = tracks["fdTOFsignal"].array()[filter]
    momentumTPC = momentumTPC[filter] 
    charge = tracks["fiCharge"].array()[filter]
    TOFBeta = tracks["fdTOFbeta"].array()[filter]
    H, xedges, yedges = np.histogram2d(momentumTPC/charge, TOFBeta, bins=(1000,1000), range=((-2000, 2000), (0, 1.2)))
    print ("done processing " + file)
    return H   

def main():
    p = Pool(8)
    histogram = sum(p.map(get_file_histogram,Common.list_files("Data")))

    fig, ax = plt.subplots()
    plt.imshow(histogram.transpose(), interpolation='nearest', origin='low', norm=colors.LogNorm(), extent= (-2000, 2000, 0, 1.2), aspect="auto")
    plt.colorbar()
    xMomentum  = np.linspace(-2000,2000, num=1000) #make a list of numbers representing momentum in MeV to use for graphing

    #draw expected values of TOF of a momentum per particle hypopthesis
    for particle in Common.particles:
        ax.plot(xMomentum,Common.velocity(np.abs(xMomentum), particle.m), label=particle.Label, linewidth=2)
    

        
    ax.set_ylim([0.2,1.2])
    ax.set_xlabel('P/z (Rigidity) [MeV/c]')
    ax.set_ylabel('beta')
    ax.set_title('TOF Signal, # of tracks: '+ str(histogram.sum()))
    plt.legend()
    plt.show()

    
if __name__ == "__main__":
    main()