# %%

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import AnimPlayer
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

#%%

folder = "" #"data/10/"

df_a = pd.read_csv(folder+"file_a.dat",comment="#",sep=",",header=None)
df_alpha = pd.read_csv(folder+"file_alpha.dat",comment="#",sep=",",header=None)
df_psi = pd.read_csv(folder+"file_psi.dat",comment="#",sep=",",header=None)
df_pi = pd.read_csv(folder+"file_pi.dat",comment="#",sep=",",header=None)

ts = df_a.iloc[1:,0].to_numpy().astype(np.float64)
r = df_a.iloc[0,1:].to_numpy().astype(np.float64)
ahist = df_a.iloc[1:,1:].to_numpy().astype(np.float64)
alphahist = df_alpha.iloc[1:,1:].to_numpy().astype(np.float64)
psihist = df_psi.iloc[1:,1:].to_numpy().astype(np.float64)
pihist = df_pi.iloc[1:,1:].to_numpy().astype(np.float64)
#%%

fig = plt.figure(figsize=(18,10))
gs = GridSpec(2,2,figure=fig)

axalpha = fig.add_subplot(gs[0,0])
axa = fig.add_subplot(gs[0,1])
axPsi = fig.add_subplot(gs[1,0])
axPi = fig.add_subplot(gs[1, 1])

plotalpha = axalpha.plot(r,alphahist[0],
                         marker="o",markersize=2)[0]
plota = axa.plot(r, ahist[0],
    marker="o",markersize=2)[0]
plotPsi = axPsi.plot(r, psihist[0],
    marker="o",markersize=2)[0]
plotPi = axPi.plot(r, pihist[0],
    marker="o",markersize=2)[0]

def update(frame):
    fig.suptitle(r"$t=$"+str(ts[frame]))

    plotalpha.set_ydata(alphahist[frame])
    plota.set_ydata(ahist[frame])
    plotPsi.set_ydata(psihist[frame])
    plotPi.set_ydata(pihist[frame])

    axalpha.relim()
    axa.relim()
    axPsi.relim()
    axPi.relim()

    axalpha.autoscale_view()
    axa.autoscale_view()
    axPsi.autoscale_view()
    axPi.autoscale_view()

ani = AnimPlayer.Player(fig=fig, func=update,maxi=ts.size-1)
plt.show()

#%%

rs = [0.6,0.69,0.75,0.85,0.95,1.03,1.14,1.24,1.36,1.46]
a0 = [2e-3,2.1e-3,2.2e-3,2.3e-3,2.4e-3,2.5e-3,2.6e-3,2.7e-3,2.8e-3,2.9e-3]

