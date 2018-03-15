import Common
import numpy as np
import math

#class for holding data for a given gas, eg argon or neon
class TPCData:
    def __init__(self, Rho, A, I, Z, X0, X1, a, m):
        self.Rho = Rho #density
        self.A = A #molecular weigh
        self.I = I #mean absobtion potential
        self.Z = Z #Atom number
        self.X0 = X0  #???
        self.X1 = X1 #???
        Ne = Common.consts.Na*Rho*Z/A
        vp = math.sqrt(Ne*Common.consts.e**2/(math.pi*Common.consts.me))
        self.C = -(2*math.log(I/(Common.consts.h*vp))+1) #???
        self.a = a #???
        self.m = m #???
        
def delta(beta, tpcData): #part of correction for bethe block
    gamma = 1/math.sqrt(1-beta**2)
    X = math.log10(beta*gamma)
    if X < tpcData.X0:
        return 0
    if X < tpcData.X1:
        return 4.6052*X+tpcData.C+tpcData.a*(tpcData.X1-X)**tpcData.m
    return 4.6052*X+tpcData.C

# calculate expected dE/dx for a given gas
# see literature for description of the formula implemented
def Bethe(momentum, mass, tpcData): 
    beta = Common.velocity(np.abs(momentum), mass)
    beta2 = beta**2
    gamma = 1/np.sqrt(1-beta2) # []
    s = Common.consts.me/mass
    eta2 = (gamma*beta)**2
    Wmax = 2*Common.consts.me*eta2/(1+2*s*np.sqrt(1+eta2)+s**2) #[MeV]

    konst = 0.1535 #MeV*cm^2/g
    deDx = konst*tpcData.Rho*tpcData.Z/tpcData.A/beta2*(np.log(2*Common.consts.me*gamma**2*beta2*Wmax/tpcData.I**2)-2*beta2)#-delta(beta,tpcData)-2*tpcData.C/tpcData.Z) # [MeV/cm]

    #return abs value to avoid overflows giving strange vertical lines in plot
    return np.abs(deDx)

#calculate dE/dx with bethe block using a gas mixture
def multipleGasBethe(momentum, mass, gasses, fractions):
    return sum(Bethe(momentum, mass, gas)*fraction for (gas,fraction) in zip(gasses,fractions))

#declare standart values for gasses
Argon = TPCData(0.001633,39.95,0.000188,18, 1.96, 4, 0.389,2.80)
CO2 = TPCData(0.001842,18.48,  0.000085,22, 1.96, 4, 0.389,2.80)

#munually fitted parameters for each particle describing in order: y-scale, x-scale and y-offset
fitParameters = [[0.911,1.161, -0.0001186], #electron
                 [5.184,1.467, -0.01483], #Muon
                 [2.361,1.163, -0.004727], #Pion
                 [4.423,1.625, -0.01069], #Kaon
                 [11.52,2.577, -0.03177], #Proton
                 [14.86,3.052, -0.03959],  #Deuterium
                 [15.9,3.166, -0.04206]  #Tritium
                 ]
