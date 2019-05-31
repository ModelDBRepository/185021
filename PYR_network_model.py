'''
PYR network model
@author: Ferguson et al. (2015) J. Comput. Neurosci.
Figure 4, Network Size 1K 

'''

import brian_no_units
from brian import *

defaultclock.dt = 0.02*ms

seed(3)

####################################################
#intrinsic parameters

# Strongly Adapting PYR cell parameters
C = 115 * pF
vr = -61.8 * mV
vpeak = 22.6 * mV
c = -65.8 * mV
N = 1000
klow = 0.1 * nS/mV
khigh = 3.3  * nS/mV
a = 0.0012 /ms
d = 10 * pA
vt = -57.0 * mV
b = 3 * nS
Ishift= 0 * pA

# Weakly Adapting PYR cell parameters
#C = 300 * pF
#vr = -61.8 * mV
#vpeak = 22.6 * mV
#c = -65.8 * mV
#N = 10000
#klow = 0.5 * nS/mV
#khigh = 3.3  * nS/mV
#a = 0.00008 /ms
#d = 5 * pA
#vt = -57.0 * mV
#b = 3 * nS
#Ishift= -45 * pA


#######################################################
#synaptic parameters
g = 1.425 * nS
alpha = 2000 /second
Beta = 333.33 /second
s_inf =alpha/(alpha+Beta)
tau_s =1/(alpha+Beta)
Ee=-15 *mV

#applied input 
Iapp = 80*pA 
Iapp_std = 15*pA 

##########################################################

#PYR cell eqns
eqs_pyr = """
Iext  : amp
Isyn = g*(v-Ee)*s_sum : amp
s_sum : 1
k=(v<vt)*klow+(v>=vt)*khigh : (siemens/volt)
du/dt = a*(b*(v-vr)-u)            : amp
dv/dt = (k*(v-vr)*(v-vt)+Ishift+Iext-Isyn-u)/C : volt
"""

#define neuron group
PYR = NeuronGroup(N, model=eqs_pyr, reset ="v = c; u += d" , threshold="v>=vpeak")

#set excitatory drive for each neuron
PYR.Iext = randn(len(PYR))*Iapp_std + Iapp

##################################################################################################################
### PYR-PYR Synapses
S=Synapses(PYR,PYR,
       model="""ds0/dt=-(s0-s_inf)/(tau_s) :1
            ds1/dt=-s1*Beta :1
            s_tot=clip(s0,0,s1) :1""",
       pre="""s0=s1; s1=(s_inf*(1-exp(-0.001*second/tau_s))+s1*exp(-0.001*second/tau_s))*exp(0.001*Beta*second)""")

PYR.s_sum=S.s_tot

##############################################################################
#set initial conditions
S.s0=0
S.s1=0
PYR.v = rand(len(PYR))*0.01 -0.065

##############################################################################
### set connectivity to 1%
S[:,:]='(rand()<0.01)'

##########################################################################
#record membrane potential of one cell and spike times of all cells in network
PYR_v = StateMonitor(PYR,'v',record=[0])
PYR_spktimes = SpikeMonitor(PYR, record=True)

###################################################################

#run for x seconds of simulated time
duration = 2 * second  

#include neuron group, synapse group, and monitors in our simulation
net =Network(PYR,S,PYR_v,PYR_spktimes)
net.run(duration)

#####################
# plot PYR membrane potential
figure(1)
plot(PYR_v.times/ms,PYR_v[0]/mV) 
xlabel("Time (ms)")
ylabel("PYR Membrane Potential (mV)")
title('PYR network with gpyr=%.3f nS, Iapp=%d pA, IappSD=%d pA'%(g/nS,Iapp/pA,Iapp_std/pA))

#make raster plot
figure(2)
raster_plot(PYR_spktimes)
xlabel("Time (ms)")
ylabel("PYR Cell Number")
title('gpyr=%.3f nS, Iapp=%d pA, IappSD=%d pA'%(g/nS,Iapp/pA,Iapp_std/pA))

show()

