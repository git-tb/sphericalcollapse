# %%

import numpy as np
import grids as gr
import field_conditions as fc
import ADM_sph as ADMs
import integrators as it
from matplotlib import pyplot as plt
import operators as op
import constraints as csr
import functools as ft
import slicing as slc
import AH as AH
import AnimPlayer
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

# %%
# =======================================


def ADM_driver(t, u, r=None, nghost=1):
    # unpack state vector
    alpha, Dalpha, A, B, DA, DB, KA, KB, lam, phi, Psi, Pi = u

    # # impose parity conditions
    # fc.BC_parity(A, is_even=True, left_side=True, nghost=nghost)
    # fc.BC_parity(B, is_even=True, left_side=True, nghost=nghost)
    # fc.BC_parity(KA, is_even=True, left_side=True, nghost=nghost)
    # fc.BC_parity(KB, is_even=True, left_side=True, nghost=nghost)
    # fc.BC_parity(phi, is_even=True, left_side=True, nghost=nghost)
    # fc.BC_parity(Pi, is_even=True, left_side=True, nghost=nghost)
    # fc.BC_parity(alpha, is_even=True, left_side=True, nghost=nghost)

    # fc.BC_parity(DA, is_even=False, left_side=True, nghost=nghost)
    # fc.BC_parity(DB, is_even=False, left_side=True, nghost=nghost)
    # fc.BC_parity(lam, is_even=False, left_side=True, nghost=nghost)
    # fc.BC_parity(Psi, is_even=False, left_side=True, nghost=nghost)
    # fc.BC_parity(Dalpha, is_even=False, left_side=True, nghost=nghost)

    # # impose extrapolation (outflow) conditions
    # fc.BC_outflow(A, nghost=nghost, order=1)
    # fc.BC_outflow(B, nghost=nghost, order=1)
    # fc.BC_outflow(DA, nghost=nghost, order=1)
    # fc.BC_outflow(DB, nghost=nghost, order=1)
    # fc.BC_outflow(KA, nghost=nghost, order=1)
    # fc.BC_outflow(KB, nghost=nghost, order=1)
    # fc.BC_outflow(lam, nghost=nghost, order=1)
    # fc.BC_outflow(phi, nghost=nghost, order=1)
    # fc.BC_outflow(Psi, nghost=nghost, order=1)
    # fc.BC_outflow(Pi, nghost=nghost, order=1)
    # fc.BC_outflow(alpha, nghost=nghost, order=1)
    # fc.BC_outflow(Dalpha, nghost=nghost, order=1)

    # specify energy momentum variables for massless scalar field
    SA, SB, rho, jA = ADMs.ADM_matter_masslessscalar_conversion(A, B, Psi, Pi)

    # time evolution of metric variables
    dt_A, dt_B, dt_DA, dt_DB, dt_KA, dt_KB, dt_lam = ADMs.ADM_sph_sys(
        alpha, Dalpha, A, B, DA, DB, KA, KB, lam, SA, SB, rho, jA, r, nghost=nghost
    )

    # time evolution of matter fields
    dt_phi, dt_Psi, dt_Pi = ADMs.ADM_matter_masslessscalar_evolution(
        alpha, A, B, Psi, Pi, r, nghost=nghost)

    # gauge conditions
    dt_alpha, dt_Dalpha = slc.oplog(alpha, KA, KB, r, nghost=nghost)

    # impose parity conditions
    fc.BC_parity(dt_A, is_even=True, left_side=True, nghost=nghost)
    fc.BC_parity(dt_B, is_even=True, left_side=True, nghost=nghost)
    fc.BC_parity(dt_KA, is_even=True, left_side=True, nghost=nghost)
    fc.BC_parity(dt_KB, is_even=True, left_side=True, nghost=nghost)
    fc.BC_parity(dt_phi, is_even=True, left_side=True, nghost=nghost)
    fc.BC_parity(dt_Pi, is_even=True, left_side=True, nghost=nghost)
    fc.BC_parity(dt_alpha, is_even=True, left_side=True, nghost=nghost)

    fc.BC_parity(dt_DA, is_even=False, left_side=True, nghost=nghost)
    fc.BC_parity(dt_DB, is_even=False, left_side=True, nghost=nghost)
    fc.BC_parity(dt_lam, is_even=False, left_side=True, nghost=nghost)
    fc.BC_parity(dt_Psi, is_even=False, left_side=True, nghost=nghost)
    fc.BC_parity(dt_Dalpha, is_even=False, left_side=True, nghost=nghost)

    # impose extrapolation (outflow) conditions
    fc.BC_outflow(dt_A, nghost=nghost, order=1)
    fc.BC_outflow(dt_B, nghost=nghost, order=1)
    fc.BC_outflow(dt_DA, nghost=nghost, order=1)
    fc.BC_outflow(dt_DB, nghost=nghost, order=1)
    fc.BC_outflow(dt_KA, nghost=nghost, order=1)
    fc.BC_outflow(dt_KB, nghost=nghost, order=1)
    fc.BC_outflow(dt_lam, nghost=nghost, order=1)
    fc.BC_outflow(dt_phi, nghost=nghost, order=1)
    fc.BC_outflow(dt_Psi, nghost=nghost, order=1)
    fc.BC_outflow(dt_Pi, nghost=nghost, order=1)
    fc.BC_outflow(dt_alpha, nghost=nghost, order=1)
    fc.BC_outflow(dt_Dalpha, nghost=nghost, order=1)

    # return update step
    res = (
        dt_alpha,
        dt_Dalpha,
        dt_A, dt_B,
        dt_DA, dt_DB,
        dt_KA, dt_KB,
        dt_lam,
        dt_phi,
        dt_Psi, dt_Pi
    )

    return np.vstack(res)


def family1_phi(r, a=1e-4):
    return a * (r**2) * np.exp(-(r-5)**2)


def family1_Psi(r, a=1e-4):
    return a * (2 * r - (r**2) * 2 * (r-5)) * np.exp(-(r-5)**2)


# %%
# =======================================
# grid parameters
nghost = 2
Nr = 500
r_a, r_b = 0, 30

r = gr.gr_CC(Nr, r_a, r_b, nghost=nghost)
dr = r[1] - r[0]

# =======================================
# initial matter field

a0 = 3e-3
iphi = family1_phi(r, a0)
iPsi = family1_Psi(r, a0)
iPi = np.zeros_like(r)

Psifunc = ft.partial(family1_Psi, a=a0)

# initial metric fields
iA, iB, iDA, iDB, iKA, iKB, ilam = ADMs.ADM_matter_masslessscalar_initial(
    r, Psifunc, nghost=nghost
)

# initial energy-momentum tensor
iSA, iSB, irho, ijA = ADMs.ADM_matter_masslessscalar_conversion(
    iA, iB, iPsi, iPi)

# initial lapse
ialpha = np.ones_like(r)
iDalpha = np.zeros_like(r)

# impose parity conditions
fc.BC_parity(iA, is_even=True, left_side=True, nghost=nghost)
fc.BC_parity(iB, is_even=True, left_side=True, nghost=nghost)
fc.BC_parity(iKA, is_even=True, left_side=True, nghost=nghost)
fc.BC_parity(iKB, is_even=True, left_side=True, nghost=nghost)
fc.BC_parity(iphi, is_even=True, left_side=True, nghost=nghost)
fc.BC_parity(iPi, is_even=True, left_side=True, nghost=nghost)
fc.BC_parity(ialpha, is_even=True, left_side=True, nghost=nghost)

fc.BC_parity(iDA, is_even=False, left_side=True, nghost=nghost)
fc.BC_parity(iDB, is_even=False, left_side=True, nghost=nghost)
fc.BC_parity(ilam, is_even=False, left_side=True, nghost=nghost)
fc.BC_parity(iPsi, is_even=False, left_side=True, nghost=nghost)
fc.BC_parity(iDalpha, is_even=False, left_side=True, nghost=nghost)

# impose extrapolation (outflow) conditions
fc.BC_outflow(iA, nghost=nghost, order=1)
fc.BC_outflow(iB, nghost=nghost, order=1)
fc.BC_outflow(iDA, nghost=nghost, order=1)
fc.BC_outflow(iDB, nghost=nghost, order=1)
fc.BC_outflow(iKA, nghost=nghost, order=1)
fc.BC_outflow(iKB, nghost=nghost, order=1)
fc.BC_outflow(ilam, nghost=nghost, order=1)
fc.BC_outflow(iphi, nghost=nghost, order=1)
fc.BC_outflow(iPsi, nghost=nghost, order=1)
fc.BC_outflow(iPi, nghost=nghost, order=1)

# =======================================
# check constraints

iH = csr.hamiltonian_constraint_sph(
    r, iA, iKA, iKB, iDA, iDB, ilam, irho, nghost=nghost)
print("Hamiltonian constraint violation (initial): ", dr*np.sum(np.abs(iH)))
iM = csr.momentum_constraint_sph(r, iKA, iKB, iDB, ijA, nghost=nghost)
print("Momentum constraint violation (initial): ", dr*np.sum(np.abs(iM)))

# =======================================
# set up state vector
u_i = ialpha, iDalpha, iA, iB, iDA, iDB, iKA, iKB, ilam, iphi, iPsi, iPi
sys_fcn = ft.partial(ADM_driver, r=r, nghost=nghost)


# %%
# =======================================
# evolve

t_i, t_f = 0, 20
CFL = 0.1
N_t = it.N_t_from_CFL(CFL, r[1] - r[0], t_i, t_f)
dt = (t_f - t_i) / N_t

Ahist = []
Bhist = []
DAhist = []
DBhist = []
lamhist = []
KAhist = []
KBhist = []
alphahist = []
Dalphahist = []
artists = []
phihist = []
Pihist = []
Psihist = []
thist = []
Hhist = []
Mhist = []

u_c = u_i
t = t_i
ts = []
for t_idx in range(int(N_t)):
    print("t=", t)
    
    u_c = it.ev_step_ICN3(t, u_c, dt, sys_fcn=sys_fcn)
    t = t + dt
    ts.append(t)

    alpha, Dalpha, A, B, DA, DB, KA, KB, lam, phi, Psi, Pi = u_c
    SA, SB, rho, jA = ADMs.ADM_matter_masslessscalar_conversion(A, B, Psi, Pi)
    H = csr.hamiltonian_constraint_sph(
        r, A, KA, KB, DA, DB, lam, rho, nghost=nghost)
    M = csr.momentum_constraint_sph(r, KA, KB, DB, jA, nghost=nghost)
    
    if(t_idx%1==0):

        SA, SB, rho, jA = ADMs.ADM_matter_masslessscalar_conversion(A, B, Psi, Pi)
        H = csr.hamiltonian_constraint_sph(
            r, A, KA, KB, DA, DB, lam, rho, nghost=nghost)
        M = csr.momentum_constraint_sph(r, KA, KB, DB, jA, nghost=nghost)

        Hhist.append(H)
        Mhist.append(M)
        
        Ahist.append(A)
        Bhist.append(B)
        DAhist.append(DA)
        DBhist.append(DB)
        lamhist.append(lam)
        KAhist.append(KA)
        KBhist.append(KB)
        alphahist.append(alpha)
        Dalphahist.append(Dalpha)
        phihist.append(phi)
        Pihist.append(Pi)
        Psihist.append(Psi)
        thist.append(t)

    # plt.plot(r, H,
    #          color=(
    #              np.min([t / (t_f - t_i), 1]),
    #              0,
    #              1-np.min([t / (t_f - t_i), 1])))
    # plt.show()

    # Is there an apparent horizon?
    # existsAH, r_AH, A_AH = AH.findAH(r, A, B, KB, nghost=nghost)
    # if (existsAH):
    #     dr_B = gr.ghost_pad(op.D1(B, r, nghost=nghost), nghost=nghost)
    #     fc.BC_parity(dr_B, is_even=False, left_side=True, nghost=nghost)
    #     H = 1/np.sqrt(A) * (2/r + dr_B/B) - 2*KB
    #     aux = H[1:]*H[:-1]
    #     print("AH!!!!", r_AH, np.any(aux))

    if (np.any(np.isnan(alpha))):
        print("t = ", t, " | nan, abort")
        break

# plt.show()
# ts = np.array(ts)
# N = np.max(np.where(np.array(alphahist) < 1))
# plt.plot(ts[:N], alphahist[:N])
# plt.gca().set_title(r"$\alpha(t,r=0)$")
# plt.show()

# =======================================
# check constraints
falpha, fDalpha, fA, fB, fDA, fDB, fKA, fKB, flam, fphi, fPsi, fPi = u_c
fSA, fSB, frho, fjA = ADMs.ADM_matter_masslessscalar_conversion(
    fA, fB, fPsi, fPi)

fH = csr.hamiltonian_constraint_sph(
    r, fA, fKA, fKB, fDA, fDB, flam, frho, nghost=nghost)
print("Hamiltonian constraint violation (final): ", dr*np.sum(np.abs(fH)))
fM = csr.momentum_constraint_sph(r, fKA, fKB, fDB, fjA, nghost=nghost)
print("Momentum constraint violation (final): ", dr*np.sum(np.abs(fM)))
# %%

fig = plt.figure(figsize=(18,10))
gs = GridSpec(3,5,figure=fig)

axalpha = fig.add_subplot(gs[0,:2])
axphi = fig.add_subplot(gs[1,:2])
axPsi = fig.add_subplot(gs[2,0])
axPi = fig.add_subplot(gs[2, 1])

axA = fig.add_subplot(gs[0,2])
axB = fig.add_subplot(gs[0,3])
axDA = fig.add_subplot(gs[1,2])
axDB = fig.add_subplot(gs[1,3])
axKA = fig.add_subplot(gs[2,2])
axKB = fig.add_subplot(gs[2,3])
axlam = fig.add_subplot(gs[0,4])
axH = fig.add_subplot(gs[1,4])
axM = fig.add_subplot(gs[2,4])

plotalpha = axalpha.plot(r,alphahist[0],
                         marker="o",markersize=2)[0]
plotA = axA.plot(r, Ahist[0],
    marker="o",markersize=2)[0]
plotB = axB.plot(r, Bhist[0],
    marker="o",markersize=2)[0]
plotDA = axDA.plot(r, DAhist[0],
    marker="o",markersize=2)[0]
plotDB = axDB.plot(r, DBhist[0],
    marker="o",markersize=2)[0]
plotKA = axKA.plot(r, KAhist[0],
    marker="o",markersize=2)[0]
plotKB = axKB.plot(r, KBhist[0],
    marker="o",markersize=2)[0]
plotlam = axlam.plot(r, lamhist[0],
    marker="o",markersize=2)[0]
plotPsi = axPsi.plot(r, Psihist[0],
    marker="o",markersize=2)[0]
plotPi = axPi.plot(r, Pihist[0],
    marker="o",markersize=2)[0]
plotphi = axphi.plot(r, phihist[0],
    marker="o",markersize=2)[0]
plotH = axH.plot(r, Hhist[0],
    marker="o",markersize=2)[0]
plotM = axM.plot(r, Mhist[0],
    marker="o",markersize=2)[0]

axalpha.set_title(r"$\alpha$")
axA.set_title(r"$A$")
axB.set_title(r"$B$")
axDA.set_title(r"$DA$")
axDB.set_title(r"$DB$")
axKA.set_title(r"$KA$")
axKB.set_title(r"$KB$")
axlam.set_title(r"$\lambda$")
axPsi.set_title(r"$\Psi$")
axPi.set_title(r"$\Pi$")
axphi.set_title(r"$\phi$")
axM.set_title(r"$M$")
axH.set_title(r"$H$")

def update(frame):
    fig.suptitle(r"$t=$"+str(thist[frame]))

    plotalpha.set_ydata(alphahist[frame])
    plotA.set_ydata(Ahist[frame])
    plotB.set_ydata(Bhist[frame])
    plotDA.set_ydata(DAhist[frame])
    plotDB.set_ydata(DBhist[frame])
    plotKA.set_ydata(KAhist[frame])
    plotKB.set_ydata(KBhist[frame])
    plotlam.set_ydata(lamhist[frame])
    plotPsi.set_ydata(Psihist[frame])
    plotPi.set_ydata(Pihist[frame])
    plotphi.set_ydata(phihist[frame])
    plotH.set_ydata(Hhist[frame])
    plotM.set_ydata(Mhist[frame])

    axalpha.relim()
    axA.relim()
    axB.relim()
    axDA.relim()
    axDB.relim()
    axKA.relim()
    axKB.relim()
    axlam.relim()
    axPsi.relim()
    axPi.relim()
    axphi.relim()
    axH.relim()
    axM.relim()

    axalpha.autoscale_view()
    axA.autoscale_view()
    axB.autoscale_view()
    axDA.autoscale_view()
    axDB.autoscale_view()
    axKA.autoscale_view()
    axKB.autoscale_view()
    axlam.autoscale_view()
    axPsi.autoscale_view()
    axPi.autoscale_view()
    axphi.autoscale_view()
    axH.autoscale_view()
    axM.autoscale_view()


ani = AnimPlayer.Player(fig=fig, func=update,maxi=len(Ahist)-1)
plt.show()


# %%
