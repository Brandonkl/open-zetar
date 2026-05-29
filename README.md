# Welcome to open-ζr!
open-ζr is an open-source code for simulating Beris-Edwards nematohydrodynamics to study active liquid crystals. It is developed by Brandon Klein and Daniel A. Beller at the Johns Hopkins University Department of Physics and Astronomy. It is an active counterpart to open-Qmin.

open-ζr is a *graph*-based code that is extendable to any number of spatial dimensions and can be easily modified to couple different fields and equations of motion. It relies on general tensor fields over space.

*This work is in progress, and bugs may occur. Please email bklein6@jh.edu for any questions.*

## Files
Currently, open-ζr supports active and passive two and three dimensional nematic liquid crystal simulations using the equations of motion popularized by Beris and Edwards. 

*As of v1.0, only periodic boundaries are implemented, more coming soon*

## Usage
- Jupyter Notebook and Python source files are provided for simulation code. 
- By default, the visualizer is automatically called after the simulation. Toggle on/of with -plot 1/0 flag.
- Install requirements.txt.
- Call python source files as:
```BE_NS_[2/3]D.py param_file -plot 1/0```
    
