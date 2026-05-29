import os
import numpy as np
import matplotlib.pyplot as plt
import numba as nb
import json
import subprocess
import sys
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams['font.size'] = 8

run_directory = sys.argv[1]
parameters = json.load(open(f"{run_directory}/{run_directory}.json"))
Lx = Ly = int(parameters["System Size"])
Z = float(parameters["Active Stress"])
A = float(parameters["DeGennes Constant"])
K = float(parameters["Elasticity"])
ss_threshold = (Z**2 + K**2 + A**2) / (100*K**2) # 1 + 1/als^4 + 1/ncl^4

bounds = []
for x in range(Lx):
    for y in range(Ly):
        bounds.append((x, y))
bounds = np.array(bounds)

@nb.njit(parallel=True, fastmath=True, nogil=True)
def get_ss_w(Q, U, ss, w, bounds):
    """ calculate saddle-splay and vorticity """ 
    Lx, Ly = Q.shape[:2]
    for point in nb.prange(len(bounds)):
        x,y = bounds[point,0], bounds[point,1]
        xup = (x + 1) % Lx
        xdn = (x - 1)
        yup = (y + 1) % Ly
        ydn = (y - 1) 
        twice_dxQxx = Q[xup,y,0] - Q[xdn,y,0]
        twice_dxQxy = Q[xup,y,1] - Q[xdn,y,1]
        twice_dyQxx = Q[x,yup,0] - Q[x,ydn,0]
        twice_dyQxy = Q[x,yup,1] - Q[x,ydn,1]
        twice_dxUy = U[xup,y,1] - U[xdn,y,1]
        twice_dyUx = U[x,yup,0] - U[x,ydn,0]
        
        ss[x,y] = twice_dxQxy*twice_dyQxx - twice_dxQxx*twice_dyQxy 
        w[x,y] = 0.5*(twice_dxUy-twice_dyUx)     

def count(ss,c,locs, Lx,Ly):

    def cluster(ss,c,sign,locs, x,y):
        x = x % Lx
        y = y % Ly
        if c[x,y] == 0 and sign*ss[x,y] > 0 :
            c[x,y] = sign
            locs[-1].append((x,y))
            return cluster(ss,c,sign,locs,x+1,y) + cluster(ss,c,sign,locs,x-1,y) + cluster(ss,c,sign,locs,x,y-1) + cluster(ss,c,sign,locs,x,y+1) + cluster(ss,c,sign,locs,x+1,y-1) + cluster(ss,c,sign,locs,x-1,y+1) + cluster(ss,c,sign,locs,x-1,y-1) + cluster(ss,c,sign,locs,x+1,y+1)
        return 0

    charges = []
    for x in range(Lx):
        for y in range(Ly):
            if ss[x,y] == 0: continue
            elif c[x,y] == 0:
                charges.append(-int(ss[x,y]/abs(ss[x,y])))
                locs.append([])
                cluster(ss,c,ss[x,y],locs,x,y)
    return charges

def avgLoc(s):
    (xavg, yavg) = (0,0)
    if 0 in s[:,0] and Lx-1 in s[:,0]: s[:,0] += Lx * (s[:,0] < Lx/2)
    if 0 in s[:,1] and Ly-1 in s[:,1]: s[:,1] += Ly * (s[:,1] < Ly/2)
    for p in s: (xavg, yavg) = np.add((xavg,yavg),p)
    return (xavg/len(s) % Lx, yavg/len(s) % Ly)

def n_from_Q(Q):
    """Extract nematic director from Q-tensor"""
    Qxx, Qxy = Q[:,:,0], Q[:,:,1]
    S = 2 * np.sqrt(Qxx**2 + Qxy**2)
    nx = np.sqrt(np.divide(Qxx, S, where=S!=0) + 1/2)
    ny = np.sqrt(1 - nx*nx) * (2*(Qxy > 0) - 1)
    return nx, ny

def create_plot(Q, U, w, ss, bounds):
    X, Y = np.meshgrid(np.arange(Lx),np.arange(Ly))
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plots_dict = {
        "ax1" : ax1,
        "pd1" : ax1.scatter([], [], color="#9B52FB", s=20, edgecolors='black', linewidths=.5),
        "nd1" : ax1.scatter([], [], color="#61FA96", marker='^', s=20, edgecolors='black', linewidths=.5),

        "ax2" : ax2,
        "pd2" : ax2.scatter([], [], color="#9B52FB", s=20, edgecolors='black', linewidths=.5),
        "nd2" : ax2.scatter([], [], color="#61FA96", marker='^', s=20, edgecolors='black', linewidths=.5)
    }

    # quiver plot for director field    
    nres = Lx // 40
    n_quiver_scale = 0.85 * Lx / nres
    plots_dict['nres'] = nres
    nx, ny = n_from_Q(Q) # get nematic director and degree of order         
    plots_dict['n_quiv'] = ax1.quiver(
        X.T[::nres,::nres], Y.T[::nres,::nres],
        nx[::nres,::nres], ny[::nres,::nres],
        headwidth=0, 
        scale=n_quiver_scale,
        pivot='middle', 
        angles='xy'
    )
    plots_dict['ss'] = ax1.imshow(ss.T, cmap = "PRGn", vmin=-2*ss_threshold, vmax=2*ss_threshold, origin='lower')
    ax1.set_aspect(Ly/Lx)
    ax1.set_title('Nematic Director Field')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_frame_on(False)
    cb1 = fig.colorbar(plots_dict['ss'], ax=ax1, fraction = 0.047, pad=0.01) # slop magic numbers
    cb1.ax.set_title(r"$SS$")

    # quiver plot for flow field    
    ures = Lx // 20
    u_quiver_scale = Z / 2
    plots_dict['ures'] = ures
    ux, uy = U[:,:,0], U[:,:,1]     
    plots_dict['u_quiv'] = ax2.quiver(
        X.T[::ures,::ures], Y.T[::ures,::ures],
        ux[::ures,::ures], uy[::ures,::ures],
        scale=u_quiver_scale,
        pivot='middle', 
        angles='xy'
    )
    plots_dict['w'] = ax2.imshow(w.T, cmap = "seismic", vmin=-Z / 200, vmax=Z / 200, origin='lower')
    ax2.set_aspect(Ly/Lx)
    ax2.set_title('Velocity Field')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_frame_on(False)
    cb2 = fig.colorbar(plots_dict['w'], ax=ax2, fraction = 0.047, pad=0.01)
    cb2.ax.set_title(r"$\omega$")
    

    return fig, plots_dict

def update_plot(fig, plots_dict, Q, U, w, ss, bounds, step, savefileprefix):
    nres = plots_dict['nres']
    ures = plots_dict['ures']
    nx, ny = n_from_Q(Q) 
    ux, uy = U[:,:,0], U[:,:,1]         
    get_ss_w(Q,U,ss,w,bounds)

    defectsp = np.ma.masked_where(ss < -ss_threshold, ss).mask * ss
    cp = np.zeros_like(defectsp)
    locsp = []
    count(defectsp,cp,locsp,Lx,Ly)
    alp = [avgLoc(np.array(l)) for l in locsp]
    plots_dict['pd1'].set_offsets(alp if len(locsp) else np.empty((0, 2)))
    plots_dict['pd2'].set_offsets(alp if len(locsp) else np.empty((0, 2)))
       
    defectsn = np.ma.masked_where(ss > ss_threshold, ss).mask * ss
    cn = np.zeros_like(defectsn)
    locsn = []
    count(defectsn,cn,locsn,Lx,Ly)
    aln = [avgLoc(np.array(l)) for l in locsn]
    plots_dict['nd1'].set_offsets(aln if len(locsn) else np.empty((0, 2)))
    plots_dict['nd2'].set_offsets(aln if len(locsn) else np.empty((0, 2)))

    plots_dict['n_quiv'].set_UVC(nx[::nres,::nres], ny[::nres,::nres]) 
    plots_dict['u_quiv'].set_UVC(ux[::ures,::ures], uy[::ures,::ures]) 
    plots_dict['w'].set_data(w.T)
    plots_dict['ss'].set_data(ss.T)
    plt.close()
    fig.savefig(savefileprefix + f'{step:10d}.png'.replace(' ','0'), bbox_inches='tight', dpi=500)

subprocess.run(f"mkdir -p {run_directory}/Images/", shell=True)
files = sorted(os.listdir(run_directory+"/data/"))
step = 0
Q = np.zeros((Lx, Ly, 2))
U = np.zeros((Lx, Ly, 2))
ss = np.zeros((Lx, Ly))
w = np.zeros((Lx, Ly))
fig, plots_dict = create_plot(Q, U, w, ss, bounds)

for file in files[1:]:

    step += 1
    Q = np.load(run_directory+"/data/"+file)['Q'].reshape((Lx, Ly, 2))
    U = np.load(run_directory+"/data/"+file)['U'].reshape((Lx, Ly, 2))
    update_plot(fig, plots_dict, Q, U, w, ss, bounds, step, savefileprefix=f"{run_directory}/Images/")

subprocess.run(f"ffmpeg -framerate 15 -pattern_type glob -y -i '{run_directory}/Images/*.png'   -c:v libx264 {run_directory}/{run_directory}.mp4", shell=True)
subprocess.run(f"rm -r {run_directory}/Images", shell=True)
