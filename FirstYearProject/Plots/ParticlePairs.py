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
    yValues = np.where(TOFBeta == -900, 0, TOFBeta - Common.velocity(momentumTPC,particle.m)) # TOF diff where TOF exists
    dist = np.sqrt((xValues*25)**2+yValues**2)
    return dist


def get_masses_file(file):
    np.seterr(all="ignore")
    tracks = uproot.open(file)["filterTracks"]["fTreeOutput"]
    TOFsignal = tracks["fdTOFsignal"].array()
    momentumTPC = tracks["fdMomentumTPC"].array() * 1000
    TOFfilter = np.logical_and.reduce((tracks["fbTPC"].array(),tracks["fbTOF"].array(),TOFsignal<40000,TOFsignal > 11000))
    betheFilter = np.logical_and.reduce((tracks["fbTPC"].array(), momentumTPC < 500))
    filter = np.logical_or(TOFfilter,betheFilter)

    charge = tracks["fiCharge"].array()[filter]
    TOFBeta = tracks["fdTOFbeta"].array()[filter]
    TOFsignal = TOFsignal[filter]
    momentumTPC = momentumTPC[filter]
    momentumGlobal = tracks["fdMomentum"].array()[filter]*1000
    dedx = tracks["fdTPCsignal"].array()[filter]/17000
    EventNumber = tracks["fiEventNumber"].array()[filter]
    Px = tracks["fdPx"].array()[filter]*1000
    Py = tracks["fdPy"].array()[filter]*1000
    Pz = tracks["fdPz"].array()[filter]*1000
    
    #evaluate hypophesies against the distance 0,01 to only get the "best" tracks
    #distances = np.vstack(np.concatenate(([calcDist(x,TOFBeta,momentumTPC,dedx) for x in range(7)],[np.ones(momentumTPC.size)*5]))) 
    distances = np.vstack([calcDist(x,TOFBeta,momentumTPC,dedx) for x in range(2,7)]) 
    #pick the most likely one for each track
    particleTypes = np.argmin(distances,axis=0)+2

    kaons = np.where(particleTypes == 3)[0]
    batchSplit = np.where(np.ediff1d(EventNumber[kaons], to_begin=0) != 0)[0]
    indexes = np.split(kaons,batchSplit)

    masses = []
    for collision  in indexes:
        particleA, particleB = np.meshgrid(collision[0:-1],collision[1:])
        particleA = particleA.flatten()
        particleB = particleB.flatten()

        E = np.sqrt(Common.particles[3].m**2+momentumGlobal[particleA]**2) + np.sqrt(Common.particles[3].m**2+momentumGlobal[particleB]**2)
        P2 = (Px[particleA]+Px[particleB])*(Py[particleA]+Py[particleB])*(Pz[particleA]+Pz[particleB])
        masses.append(np.sqrt(E**2-P2))

    print ("done processing " + file)
    masses = np.concatenate(masses)
    return masses[~np.isnan(masses)]


def main():
    np.seterr(all="ignore")
    p = Pool(4)
    masses = np.concatenate(p.map(get_masses_file,Common.list_files("Data")))

    fig, ax = plt.subplots()
    plt.hist(masses[masses<10000], bins = 900, histtype = "step", range = (600,1500))

    ax.set_title('Mass of mother particle of (kaon - kaon)')
    ax.set_xlabel('Reconstructed mass [MeV]')
    ax.set_ylabel('# of particle pairs per 1 MeV bin')
    ax.set_xlim([600,1500])

    plt.show()
    
if __name__ == "__main__":
    main()