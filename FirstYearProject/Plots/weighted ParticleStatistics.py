import math
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import uproot
import Common
import BetheBlock
import multiprocessing as mp
from multiprocessing import Pool


# calculates the weighted distance from the expected values for a given particle. Eg. if particleNr=0, calculates expected TOF and dEdx, and evaluates the difference.
# The particle hypophethis which for a given track gives the smallest "distance" is considered the best hypophesis
def calcDist(particleNr,TOFBeta, momentumTPC, dedx):
    fit = BetheBlock.fitParameters[particleNr]
    particle = Common.particles[particleNr] #particle hypophesis
    xValues = dedx-(fit[2]+fit[0]*BetheBlock.multipleGasBethe(fit[1]*momentumTPC, particle.m, [BetheBlock.Argon, BetheBlock.CO2], [0.88,0.12])) # dEdx diff
    yValues = TOFBeta - Common.velocity(momentumTPC,particle.m) # TOF diff
    dist = np.sqrt((xValues*25)**2+yValues**2)
    return dist


def get_file_histogram(file):
    np.seterr(all="ignore")
    tracks = uproot.open(file)["filterTracks"]["fTreeOutput"]
    TOFsignal = tracks["fdTOFsignal"].array()
    momentumTPC = tracks["fdMomentumTPC"].array()*1000
    psudorapidity = tracks["fdEta"].array()
    filter = np.logical_and.reduce((tracks["fbTPC"].array(),tracks["fbTOF"].array(),TOFsignal<40000,TOFsignal > 11000, momentumTPC/np.cosh(psudorapidity) > 550))
    charge = tracks["fiCharge"].array()[filter]
    TOFBeta = tracks["fdTOFbeta"].array()[filter]
    TOFsignal = TOFsignal[filter]
    momentumTPC = momentumTPC[filter]
    dedx = tracks["fdTPCsignal"].array()[filter]/17000
    
    #evaluate hypophesies
    distances = np.vstack([calcDist(x,TOFBeta,momentumTPC,dedx) for x in range(2,7)])
    #pick the most likely one for each track
    particleTypes = np.argmin(distances,axis=0)

    particleDistributions = []
    for index, particle in enumerate(Common.particles[2:]):

        indexes = particleTypes == index
        momenta = momentumTPC[indexes]
        propabilities = (1/2)**((TOFsignal[indexes]*10**-12*np.sqrt(1-Common.velocity(momenta,particle.m))/particle.halflife))


        y,binEdges = np.histogram(momenta/charge[indexes],bins=100,range=(-1500,1500), weights=1/propabilities)

        particleDistributions.append(y)
    print ("done processing " + file)
    return particleDistributions


def main():
    np.seterr(all="ignore")
    p = Pool(8)
    returns = p.map(get_file_histogram,Common.list_files("Data"))

    fig, ax = plt.subplots()
    #plot calculated distributions
    for index, particle in enumerate(Common.particles[2:]):
        histogram = sum([i[index] for i in returns])
        plt.plot(range(-1500,1500,30),histogram,'-',label=particle.Label + " #: " + str(np.sum(histogram)))
    plt.legend()

    ax.set_title('Weighted Particle Distribution')
    ax.set_xlabel('P/z (Rigidity) [MeV/c]')
    ax.set_ylabel('# of particles/30MeV bin')
    plt.ylim(ymin=0.9)
    plt.yscale("log")

    plt.show()
    
if __name__ == "__main__":
    main()