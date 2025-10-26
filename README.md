# soSPIM-FLIM

Python code associated to "Fast volumetric fluorescence lifetime imaging of multicellular systems using single-objective light-sheet microscopy" by Dunsing-Eichenauer, Hummert et al., bioRxiv 2024.03.24.586451; doi: https://doi.org/10.1101/2024.03.24.586451. 

The code allows analyzing monoexponential fluorescence lifetime decays detected with the time-gated 512x512 SPAD array detector (https://piimaging.com/spad-512/#spad512)

Requirements:
- python 3.11.5
- scipy 1.11.1
- numpy 1.25.2
- tifffile 2023.7.18
- matplotlib 3.8.0
- scikit image 0.21.0

To run the code on the provided example data, run analyse_monoexp.py after adjusting the paths to load the IRF, background and data files.
