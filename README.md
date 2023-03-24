# studiodarwin

README.md
* **Author**: Vito Dichio
* **email**: dichio.vito@gmail.com
* **Last Modified**: 24 March 2023

Source code for Dichio, V., Zeng, H. L., & Aurell, E. (2021). Statistical Genetics and Direct Coupling Analysis in and out of Quasi-Linkage Equilibrium. Reports on Progress in Physics.** (2021).

Requirements
------------------
> Based on **FFPopSim** (Python 2.7 & C++): Zanini, F., & Neher, R. A. (2012). FFPopSim: an efficient forward simulation package for the evolution of large populations. Bioinformatics, 28(24), 3332-3333. Installation required.

> **PLM-DCA** (MATLAB & C++) for PLM inference as in in Gao, C. Y., Cecconi, F., Vulpiani, A., Zhou, H. J., & Aurell, E. (2019). DCA for genome-wide epistasis analysis: The statistical genetics perspective. Physical biology, 16(2), 026002. The source code is available at https://github.com/gaochenyi/CC-PLM. 

Description
---------------
**myevolution.py** simulates  evolution by exploiting FFPopSim. Various instantaneous and/or all-time plots can be chosen.

**crystalball.py** old version of myevolution.py, contains (optionally) the inference step. 

**evol_plotter.py** contains the source code for the plots that appear in the report.

