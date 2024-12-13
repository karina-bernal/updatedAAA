# Updated AAA 

This is an updated version of the **Automatic Analysis Architecture (AAA)** tool originally published by Malfante, Mars & Della Mura (2018) (available at [GitHub](https://github.com/malfante/AAA) and [Zenodo](https://zenodo.org/records/1216028)). We recommend checking their documentation for a complete overview of the AAA tool.

The Updated AAA (UAAA) is a versatile tool designed for building and implementing machine learning classification models to analyze continuous time series data. While initially developed for the automatic classification of seismo-volcanic signals at Popocatépetl Volcano in Mexico, it can be adapted for analyzing other types of continuous signals in various scientific and industrial applications. This enhanced version introduces improvements and new features designed to optimize analysis workflows and extend functionality.

<br>
The original peer-reviewed paper for which the UAAA workflow was developped is:

**Bernal-Manzanilla, K., Calò, M., Martínez-Jaramillo, D., Valade, S. (2024, in revision). Automated Seismo-Volcanic Event Detection Applied to Popocatépetl using Machine Learning. Journal of Volcanology and Geothermal Research. Pre-print available at SSRN: https://ssrn.com/abstract=4972730 or http://dx.doi.org/10.2139/ssrn.4972730**

## Installation  

To set up the environment for the Updated AAA (UAAA) tool, we provide a `UAAA.yml` file for creating a Conda virtual environment. You can install it using the following command:  

```bash
conda env create -f UAAA.yml
```

Alternatively, you can create the Conda environment manually and install the dependencies listed in the .yml file. This option allows for customization if needed.
For example:
```bash
conda create -n UAAA python=3.6.13
conda activate UAAA  # To activate environment
pip install numpy==1.13.3  # Can install dependencies with conda or pip
```
## Featured Updates:  
Here are the main updates featured in the UAAA workflow:
- Implementation of Principal Component Analysis (PCA) to reduce the dimensionality of the feature space.  
- Inclusion of performance metrics per class during cross-validation and testing of the classification model.  
- Introduction of a dynamic window size for analyzing continuous recordings.  
- Enhanced visualization tools for exploratory data analysis and result interpretation.
<br>
A detailed description of each update can be found in the following subsections.


## Enhanced visualization

We leverage the SWARM software for results' visualization. SWARM is a free software available at the USGS's [Volcano Hazard Program](https://volcanoes.usgs.gov/software/swarm/index.shtml)) for the visualization and analysis of continuous seismic data.
