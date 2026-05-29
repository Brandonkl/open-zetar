import subprocess
import numpy as np
import numba as nb
import sys
import os
import pyvista as pv

@nb.njit(parallel=True, fastmath=True, nogil=True)
def getCoords(coords, Lx, Ly, Lz):
    for p in nb.prange(Lx * Ly * Lz):
        x = p // (Ly * Lz)
        y = (p % (Ly * Lz)) // Lz
        z = (p % (Ly * Lz)) % Lz
        xup = ((x + 1) % Lx) * (Ly * Lz) + y * Lz + z
        xdn = ((x - 1) % Lx) * (Ly * Lz) + y * Lz + z
        yup = x * (Ly * Lz) + ((y + 1) % Ly) * Lz + z
        ydn = x * (Ly * Lz) + ((y - 1) % Ly) * Lz + z
        zup = x * (Ly * Lz) + y * Lz + ((z + 1) % Lz)
        zdn = x * (Ly * Lz) + y * Lz + ((z - 1) % Lz)
        coords[p] =[x, y, z, xup, xdn, yup, ydn, zup, zdn]

@nb.njit(fastmath=True)
def QvecToMat(Qvec):
    Qxx, Qxy, Qxz, Qyy, Qyz = Qvec
    return np.array([[Qxx, Qxy, Qxz],[Qxy, Qyy, Qyz],[Qxz, Qyz, -Qxx-Qyy]])

@nb.njit(parallel=True, fastmath=True, nogil=True)
def markDefects(Q, defects, cos_beta, coords, lc_guvilk):
    for p in nb.prange(Q.shape[0]):

        S, _ = np.linalg.eigh(QvecToMat(Q[p]))
        if S[-1] < .45: 
            x, y, z, xup, xdn, yup, ydn, zup, zdn = coords[p]
            defects[x, y, z] = 1
            
            GradQ = np.zeros((3, 3, 3))
            GradQ[0] = QvecToMat(Q[xup] - Q[xdn])
            GradQ[1] = QvecToMat(Q[yup] - Q[ydn])
            GradQ[2] = QvecToMat(Q[zup] - Q[zdn])
            GradQ *= 0.5
            # D_{gi} = e_{guv}e_{ilk}d_lQ_{ua}d_kQ_{va}
            D = np.zeros((3, 3))
            for g in range(3):
                for i in range(3):
                    for u in range(3):
                        for v in range(3):
                            for l in range(3):
                                for k in range(3):
                                    D[g, i] += (
                                        lc_guvilk[g,u,v,i,l,k]
                                        * np.sum(GradQ[l, u, :] * GradQ[k, v, :]))
            norm = np.linalg.norm(D)
            if norm > 0: cos_beta[x, y, z] = max(min(np.trace(D) / norm, 1), -1)

if __name__ == "__main__":
    run_directory = sys.argv[1]
    pv.global_theme.allow_empty_mesh = True
    nb.set_num_threads(os.cpu_count() - 1)
    subprocess.run(f"mkdir -p {run_directory}/Images/", shell=True)
    lc = np.zeros((3, 3, 3))
    lc[0, 1, 2] = lc[1, 2, 0] = lc[2, 0, 1] = 1
    lc[2, 1, 0] = lc[1, 0, 2] = lc[0, 2, 1] = -1
    lc_guvilk = np.einsum("guv,ilk->guvilk", lc, lc)
    light = pv.Light(position=(0, 1, 3), show_actor=True, positional=True,
            cone_angle=30, exponent=20, intensity=1.5)
    light.positional = True

    Lx = Ly = Lz = int(sys.argv[2])
    coords = np.zeros((Lx * Ly * Lz, 9), dtype=np.int64)
    getCoords(coords, Lx, Ly, Lz)
    for file in sorted(os.listdir(run_directory+"/Q/"))[4:]:
        
        time_stamp = file[:-4]
        defects = np.zeros((Lx, Ly, Lz))
        cos_beta = np.zeros(defects.shape)
        
        markDefects(np.load(run_directory+"/Q/"+file)['Q'], defects, cos_beta, coords, lc_guvilk)
        pl = pv.Plotter(off_screen=True)
        pl.add_mesh(pv.PolyData(np.argwhere(defects > 0.5)), scalars=cos_beta[defects > 0.5], cmap="jet", ambient=0, specular=0.5, diffuse=0.5, smooth_shading=True)
        pl.add_light(light)
        pl.set_background("black")
        pl.screenshot(f"{run_directory}/Images/{time_stamp}.png")
    subprocess.run(f"ffmpeg -framerate 15 -pattern_type glob -y -i '{run_directory}/Images/*.png' -c:v libx264 -pix_fmt yuv420p {run_directory}/{run_directory}.mp4", shell=True)
