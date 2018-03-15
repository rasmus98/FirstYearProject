import math
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import uproot
import Common
import BetheBlock
from multiprocessing import Pool


def get_file_histogram(file):
    np.seterr(all="ignore")
    tracks = uproot.open(file)["filterTracks"]["fTreeOutput"]
    momentumTPC = tracks["fdMomentumTPC"].array()*1000
    filter = np.logical_and.reduce((tracks["fbTPC"].array(),tracks["fbTOF"].array(),tracks["fdTOFsignal"].array()<(25+12)*1000,momentumTPC>1000, momentumTPC<1200))
    charge = tracks["fiCharge"].array()[filter]
    TOFBeta = tracks["fdTOFbeta"].array()[filter]
    momentumTPC = tracks["fdMomentumTPC"].array()[filter]*1000
    dedx = tracks["fdTPCsignal"].array()[filter]/17000
    histograms = []

    for particleNr in range(2,7):
        fit = BetheBlock.fitParameters[particleNr]
        particle = Common.particles[particleNr]
        xValues = dedx-(fit[2]+fit[0]*BetheBlock.multipleGasBethe(fit[1]*momentumTPC, particle.m, [BetheBlock.Argon, BetheBlock.CO2], [0.88,0.12]))
        yValues = TOFBeta-Common.velocity(momentumTPC, particle.m)

        H, xedges, yedges = np.histogram2d(xValues,
               yValues, 
               bins=(1000,1000), 
               range=((-0.01, 0.01), (-0.30, 0.30)))

        histograms.append(H)
    print ("done processing " + file)
    return histograms

def main():
    np.seterr(all="ignore")
    p = Pool(8)
    returns = p.map(get_file_histogram,Common.list_files("Data"))

    # for a given particle hypophesies, calculate difference between expected and actual TOF and dEdx
    for particleNr in range(2,7):
        histogram = sum([i[particleNr-2] for i in returns])
        fig, ax = plt.subplots()
        plt.imshow(histogram.transpose(), interpolation='nearest', norm=colors.LogNorm(), extent= (-0.01, 0.01, -0.30, 0.30), aspect="auto")
        plt.colorbar()
        
        ax.grid(color='r', linestyle='-', linewidth=1)
        ax.set_xlabel('dE/dx-<dE/dx>'+Common.particles[particleNr].Label)
        ax.set_ylabel('beta-<beta>'+Common.particles[particleNr].Label)
        ax.set_title('PID, # of tracks: '+ str(histogram.sum()))

        plt.show()

if __name__ == "__main__":
    main()