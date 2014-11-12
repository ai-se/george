####Interpolation
|Size(k*nasa93)|Median-knn|IQR-knn|Median-clusterk1|IQR-clusterk1|Median-CART|IQR-CART
|:---|:---|:---|:---|:---|:---|:---|
|1|1.98|8.52|0.57|0.79|0.39|0.66|
|2|4.82|17.89|0.65|1.13|0.48|0.58|
|4|3.60|10.27|0.62|1.16|0.41|0.60|
|8|5.61|20.74|0.54|0.76|0.43|0.62|
|16|4.40|13.55|0.48|0.74|0.36|0.59|

#####Crashes For nasa93 * 32 since all the records in a cluster are same and dist(west,east) = 0(divide by 0 error)


###Extrapolation
|Size(k*nasa93)|Median-knn|IQR-knn|Median-clusterk1|IQR-clusterk1|Median-CART|IQR-CART
|:---|:---|:---|:---|:---|:---|:---|
|1|1.99|8.52|0.57|0.79|0.39|0.66|
|2|3.77|8.86|0.61|0.86|0.58|0.76|
|4|2.67|5.04|0.61|0.73|0.42|0.64|
|8|3.34|8.72|0.71|1.02|0.39|0.69|
|16|4.63|11.16|0.54|0.82|0.43|0.79|
|32|5.26|17.12|0.60|0.92|0.47|0.69|

#####Crashes For nasa93 * 64 since all the records in a cluster are same and dist(west,east) = 0(divide by 0 error)
