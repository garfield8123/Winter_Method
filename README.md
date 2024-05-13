# **Winter Method**
A statiscal model reference of winter's method that can calcualte seasonalized demand, de-seasonalized demand, seasonality, average seasonality, level, trend, forecast error, MAPE (Mean Absolute Percent Error) error, bias error, tracking signal, best smoothing constnat, and as well as forecasting value

``` Python
winters_method(dataframe, Sequence_Column_Name, Fleet_Column_Name, demand_Column_Name, output_Filled_CSV)
```
- dataframe = pandas.dataframe that includes all the data
- Sequence_Column_Name = String column name of the sesonalized column you would like to use
- Fleet_Column_Name = String column name of the column you are ok with having copies 
- demand_Column_Name = String column name of the column you are looking to forecast
- output_Filled_CSV = String name for the output of all the data should be located


> [!IMPORTANT]
> This code will proprage a random selection of the given values within the same seasonality period to ensure the amount of data per seasonality is equilvalent to one another. 
