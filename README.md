# Welcome to open-ζr!
open-ζr is a CPU-parallelized, open-source code for simulating and visualizing passive and active nematic liquid crystals in two and three dimensions. It is developed by Brandon Klein and Daniel A. Beller at the Johns Hopkins University Department of Physics and Astronomy. It is an active counterpart to open-Qmin.

*This work is in progress, and bugs may occur. Please email bklein6@jh.edu for any questions.*

## Files
Currently, open-ζr supports active and passive two and three dimensional nematic liquid crystal simulations using the equations of motion popularized by Beris and Edwards. 

*As of v1.0, only periodic boundaries are implemented, more coming soon*

## Usage
- Jupyter Notebook and Python source files are provided for simulation code. 
- By default, the visualizer is automatically called after the simulation. Toggle on/of with -no_plot flag.
- Install requirements.txt manually or through ```pip install requirements.txt```
- Call python source files as:
```BE_NS_[2/3]D.py -param_file [File] [ /-no_plot]```
    
