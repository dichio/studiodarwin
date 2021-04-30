# studiodarwin
README.md

* **Author**: Vito Dichio
* **Last Modified**: 24 Apr. 2021
* 
Description
------------
Source code for Dichio, V., Zeng, H. & Aurell, E., **Statistical Genetics and Direct Coupling Analysis beyond Quasi-Linkage Equilibrium** (2021).

Requirements
------------------
	- Based on **FFPopSim** (Python 2.7 & C++): Zanini, F., & Neher, R. A. (2012). FFPopSim: an efficient forward simulation package for the evolution of large populations. Bioinformatics, 28(24), 3332-3333. https://doi.org/10.1093/bioinformatics/bts633. Installation required.
	- PLM-DCA (MATLAB & C++) for PLM inference as in in Gao, C. Y., Cecconi, F., Vulpiani, A., Zhou, H. J., & Aurell, E. (2019). DCA for genome-wide epistasis analysis: The statistical genetics perspective. Physical biology, 16(2), 026002. The source code is available at https://github.com/gaochenyi/CC-PLM. 

Description
---------------
**crystalball.py** simulates  evolution by exploiting FFPopSim and optionally can infer from simulated data (naive Mean Field or Pseudo-Likelihood Maximization for NS inference or GC inference). Various instantaneous and/or all-time plots can be chosen.
**crystalball_iter.py** is able to classify NRC/QLE phases automatically.
**crystalball_ls.py** is adapted for long-term simulations eg study of the escape times from the QLE/NRC phases.

![image](https://user-images.githubusercontent.com/79842912/116705088-779afa80-a9cc-11eb-9c69-e101d969c795.png)
