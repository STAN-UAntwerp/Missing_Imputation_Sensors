# Missing Value Imputation of Wireless Sensor Data for Environmental Monitoring

### Authors: Thomas Decorte, Steven Mortier, Jonas J. Lembrechts, Filip J. R. Meysman, Steven Latré, Erik Mannens, and Tim Verdonck

## Abstract
Over the past few years, the scale of sensor networks has greatly expanded. This generates 1
extended spatio-temporal datasets, which form a crucial information resource in numerous fields, 2
ranging from sports and healthcare to environmental science and surveillance. Unfortunately, these 3
datasets often contain missing values, due to systematic or inadvertent sensor misoperation. This 4
incompleteness hampers the subsequent data analysis, yet addressing these missing observations 5
forms a challenging problem. This is especially the case when both the temporal correlation of 6
timestamps within a single sensor as well as the spatial correlation between sensors are important. 7
Here, we compare 12 imputation methods to complete the missing values in a dataset originating 8
from large-scale environmental monitoring. As part of a large citizen science project, IoT-based 9
microclimate sensors were deployed for 6 months in 4400 gardens across the region of Flanders, 10
generating 15 minute recordings of temperature and soil moisture. Methods based on spatial recovery 11
as well as time-based imputation were evaluated, including Spline Interpolation, MissForest, MICE, 12
MCMC, M-RNN, BRITS, and others. The performance of these imputation methods was evaluated for 13
different proportions of missing data (ranging from 10% to 50%), as well as a realistic missing value 14
scenario. Techniques leveraging the spatial features of the data tend to outperform the time-based 15
methods, with matrix completion techniques providing the best performance. Our results hence 16
provide a tool to maximize the benefit from costly, large-scale environmental monitoring efforts.

For more information see: INSERT PAPER URL

## Dependencies & Code

The code contains the preprocessing steps of the data as well as the various methods applied in the paper alongside the general setup and hyperparemeter tuning leveraged. 
The code was implemented in Python 3.11. The following packages are needed for running the code:
- numpy==1.23.3
- pandas==1.5.0
- scikit-learn==1.1.2
- scipy==1.9.1
- sklearn==0.0
- statsmodels==0.13.2

## Datasets
The datasets showcased in the paper are made available on the SoilTemp project (https://www.soiltempproject.com/) of the University of Antwerp.

## Example usage

To execute an example of the procedure based on the above dataset run the following line:
```
 python allmethods.py
```
