# Missing Value Imputation of Wireless Sensor Data for Environmental Monitoring

This repository contains the code accompanying the publication titled *Missing Value Imputation of Wireless Sensor Data for Environmental Monitoring*. You can find the paper as open access [here](https://www.mdpi.com/1424-8220/24/8/2416)

### Authors: Thomas Decorte, Steven Mortier, Jonas J. Lembrechts, Filip J. R. Meysman, Steven Latré, Erik Mannens, and Tim Verdonck

## Abstract
Over the past few years, the scale of sensor networks has greatly expanded. This generates
extended spatio-temporal datasets, which form a crucial information resource in numerous fields,
ranging from sports and healthcare to environmental science and surveillance. Unfortunately, these
datasets often contain missing values, due to systematic or inadvertent sensor misoperation. This
incompleteness hampers the subsequent data analysis, yet addressing these missing observations
forms a challenging problem. This is especially the case when both the temporal correlation of
timestamps within a single sensor as well as the spatial correlation between sensors are important.
Here, we compare 12 imputation methods to complete the missing values in a dataset originating
from large-scale environmental monitoring. As part of a large citizen science project, IoT-based
microclimate sensors were deployed for 6 months in 4400 gardens across the region of Flanders,
generating 15 minute recordings of temperature and soil moisture. Methods based on spatial recovery
as well as time-based imputation were evaluated, including Spline Interpolation, MissForest, MICE,
MCMC, M-RNN, BRITS, and others. The performance of these imputation methods was evaluated for
different proportions of missing data (ranging from 10% to 50%), as well as a realistic missing value
scenario. Techniques leveraging the spatial features of the data tend to outperform the time-based
methods, with matrix completion techniques providing the best performance. Our results hence
provide a tool to maximize the benefit from costly, large-scale environmental monitoring efforts.

## Code

The code contains the preprocessing steps of the data (data cleaning, distance calculation of sensors etc.) as well as the various methods applied in the paper alongside the general setup and hyperparemeter tuning leveraged. The code was implemented in Python 3.11. 

## Datasets
The datasets showcased in the paper are made available on the SoilTemp project (https://www.soiltempproject.com/) of the University of Antwerp.

## Example usage

To execute an example of the procedure based on the above dataset run the following line:
```
 python allmethods.py
```
