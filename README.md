# CATGNN_phonon_DOS
The framework of the model is shown in the figure below:</br>
![figure1](https://github.com/user-attachments/assets/169ce72d-256a-4428-9ae6-3a871d8a8ed2)


This code was used to predict phonon DOS using our developed CATGNN model which was used to produce the following work:</br>
- Al-Fahdi, M., Lin, C., Shen, C., Zhang, H., & Hu, M. Rapid prediction of phonon density
of states by crystal attention graph neural network and High-Throughput screening of candidate
substrates for wide bandgap electronic cooling. *Materials Today Physics*, **2025**, 101632. </br>
**Note**: that the correct figures 1-3 in the above work are in the corrigendum in the following [link](https://www.sciencedirect.com/science/article/pii/S2542529325000094?via%3Dihub).
- please cite the above work if you use the code

## Required Packages
the following packages are required to run the code:</br>
<code>torch=2.5.1</code></br>
<code>torch-geometric=2.6.1</code></br>
<code>torch-scatter=2.1.2</code></br>
<code>e3nn=0.5.1</code></br>
<code>Jarvis-tools=2024.10.30</code></br>
<code>scikit-learn=1.2.2</code></br>

other versions might work, but those versions were successful in running the code

## Usage
1- untar the data directory by running:</br>
<code>tar -xvzf data.tar</code></br>
2- you can edit the model parameters from the file "model_params.yaml" and the parameters should be straightforward to edit.</br>
3- you can simply run the following line to run the code:</br>
<code>python main.py</code>

