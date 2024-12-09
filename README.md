# Updated AAA

This is an updated version of the code originally published by Malfante, Mars & Della Mura (2018) (available at [GitHub](https://github.com/malfante/AAA) and [Zenodo](https://zenodo.org/records/1216028))

The updates were implemented in the classification of seismo-volcanic signals but can be extended to other applications.

The original paper for which the updated code was developped is:

**Bernal-Manzanilla, K., Calò, M., Martínez-Jaramillo, D., Valade, S. (2024, in revision). Automated Seismo-Volcanic Event Detection Applied to Popocatépetl using Machine Learning. Journal of Volcanology and Geothermal Research. Pre-print available at SSRN: https://ssrn.com/abstract=4972730 or http://dx.doi.org/10.2139/ssrn.4972730**

The updates include:
- Use of Principal Component Analysis (PCA) to reduce the dimensions of the feature space.
- Perfomance metrics per class during cross-validation and testing of the classification model.
- Use of a dynamic size window in the analysis of continuous recodings.
