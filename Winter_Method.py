
def Column_Seasonality(dataframe, Sequence_Column_Name):
    #Split main dataframe by string and place it into object callable?
    #3
    sequenceList = set(dataframe[Sequence_Column_Name].tolist())
    seasonalityData = {}
    for x in sequenceList:
        seasonality = dataframe.copy()
        seasonality = seasonality[seasonality[Sequence_Column_Name].apply(lambda y: y is x)]
        seasonalityData.update({x:seasonality})
    return seasonalityData


def Calculate_Number_Of_Periods(seasonalityData):
    NumberOfPeriods = len(seasonalityData.keys())
    return NumberOfPeriods


def Fill_Columns(dataframe, Sequence_Column_Name, seasonalityData, Fleet_Column_Name, output_Filled_CSV):
    import pandas
    lengthList = {}
    for x in seasonalityData.keys(): 
        lengthList.update({x: seasonalityData.get(x).shape[0]})
    sorted_items = sorted(lengthList.values(), reverse=True)
    sorted_dict = {k: v for v in sorted_items for k, val in lengthList.items() if val == v}
    largestLength = sorted_items[0]
    for x in sorted_dict.keys():
        count = seasonalityData.get(x).shape[0]
        if sorted_dict.get(x) != largestLength:
            copy = 1
            while count < largestLength:
                random_row = seasonalityData.get(x).sample(n=1, replace=False)  # Selecting a random row
                copied_row = random_row.copy()  # Creating a deep copy of the random row
                copied_row[Fleet_Column_Name] = "copy_" + str(copy)
                copy = copy + 1
                seasonalityData.update({x: pandas.concat([seasonalityData.get(x), copied_row], ignore_index=True)})
                count = count + 1
        dataframe = dataframe.drop_duplicates()
        dataframe = pandas.concat([dataframe, seasonalityData.get(x)], ignore_index=True)
        #filtered_training_data = filtered_training_data.drop_duplicates()
    dataframe.to_csv(output_Filled_CSV, index=False, header=True)
    return dataframe


def winters_method_Seasonalized_Demand(dataframe, NumberOfPeriods, Sequence_Column_Name, demand_Column_Name, seasonalityData):
    # run the equation =(M2+M6+2*SUM(M3:M5))/(2*L4) (Assuming NumberOfPeriods = 4)
    # L Column = Number of periods
    # M column = Demand data (Depart time in unix_timestamp)
    filtered_training_data_sorted = dataframe.sort_values(by=Sequence_Column_Name, ascending=False)
    #filtered_training_data_sorted.to_csv("output.csv", header=False, index=False)
    Sesonalized_Demand_values = {}
    seasonalized_Demand_list = []
    for x in seasonalityData.keys():
        Sesonalized_Demand_values.update({x: []})

    count = 0
    if NumberOfPeriods % 2 == 0:
        limit = NumberOfPeriods
    else:
        limit = 0
    while (count < filtered_training_data_sorted.shape[0] - limit):
        #print(count)
        first_element = filtered_training_data_sorted.iloc[int(count)]
        if NumberOfPeriods % 2 == 0:
            last_element = filtered_training_data_sorted.iloc[int(count+NumberOfPeriods)]
            between_element = filtered_training_data_sorted.iloc[int(count+1):int(count+NumberOfPeriods-1)]
            count+=1
            Sesonalized_Demand = (first_element[demand_Column_Name] + last_element[demand_Column_Name] + 2 * between_element[demand_Column_Name].sum()) / (2 * NumberOfPeriods)
        else:
            between_element = filtered_training_data_sorted.iloc[int(count):int(count+NumberOfPeriods-1)]
            count+=1
            Sesonalized_Demand = between_element[demand_Column_Name].sum() / NumberOfPeriods
        #print(type(first_element["arrive_time"]))
        seasonalized_Demand_list.append(Sesonalized_Demand)
        Sesonalized_Demand_values.get(first_element[Sequence_Column_Name]).append(Sesonalized_Demand)
    return seasonalized_Demand_list, Sesonalized_Demand_values


def winters_method_De_seasonalized_Demand(dataframe, NumberOfPeriods, seasonalityData, seasonalized_Demand_list, Sesonalized_Demand_values):
    # run the equation =INTERCEPT($O$2:O31,$N$2:N31)+N4*SLOPE($O$2:O31,$N$2:N31)
    # O Column = Seasonalized Demand Data
    # N Column = Period Number (Incremental value of 1)
    #seasonalized_Demand_list
    periodlist = [x for x in range(dataframe.shape[0])]

    if NumberOfPeriods % 2 == 0: 
        listof0 = [0 for x in range(NumberOfPeriods // 2)]
        seasonalized_Demand_list = listof0 + seasonalized_Demand_list + listof0

    De_Sesonalized_Demand_values = {}
    De_seasonalized_Demand_list = []
    for x in seasonalityData.keys():
        De_Sesonalized_Demand_values.update({x: []})


    from scipy.stats import linregress

    slope, intercept , r_value, p_value, std_err = linregress(periodlist, seasonalized_Demand_list)
    for x in range(len(periodlist)):
        De_sesaonzlied_demand = intercept + int(periodlist[x]) * slope
        De_seasonalized_Demand_list.append(De_sesaonzlied_demand)
        for key, value_list in Sesonalized_Demand_values.items():
            if seasonalized_Demand_list[x] in value_list:
                De_Sesonalized_Demand_values.get(key).append(De_sesaonzlied_demand)
    return De_seasonalized_Demand_list, De_Sesonalized_Demand_values, periodlist


def winters_method_Seasonality(dataframe, demand_Column_Name, seasonalityData, De_seasonalized_Demand_list, De_Sesonalized_Demand_values):
    # Run the equation =M2/P2
    # M Column = Demand Data (Depart time in unix timestamp)
    # P Column = De-seasonalized Demand Data 
    Demand_data = dataframe[demand_Column_Name].tolist()
    Seasonal_Factors_Values = {}
    Seasonal_Factors_List = []
    for x in seasonalityData.keys():
        Seasonal_Factors_Values.update({x: []})

    for x, y in zip(Demand_data, De_seasonalized_Demand_list):
        result = x / y
        Seasonal_Factors_List.append(result)
        for key, value_list in De_Sesonalized_Demand_values.items():
            if y in value_list:
                Seasonal_Factors_Values.get(key).append(result)
    return Seasonal_Factors_List, Seasonal_Factors_Values


def winters_method_average_Seasonality(seasonalityData, Seasonal_Factors_Values):
    # Run the equation =(Q2+Q6+Q10+Q14+Q18+Q22+Q26)/7 (Assuming NumberOfPeriods is 4) 
    # Q Column = Seasonality 
    import numpy
    Average_Seasonal_Factors_Values = {}
    for x in seasonalityData.keys():
        Average_Seasonal_Factors_Values.update({x: []})
    for x in Seasonal_Factors_Values.keys():
        Average_Seasonal_Factors_Values.get(x).append(numpy.mean(Seasonal_Factors_Values.get(x)))
    return Average_Seasonal_Factors_Values


def Initial_Level_And_Trend(De_seasonalized_Demand_list, periodlist):
    from scipy.stats import linregress
    slope, intercept , r_value, p_value, std_err = linregress(De_seasonalized_Demand_list, periodlist)
    initial_level = intercept
    initial_trend = slope
    return initial_level, initial_trend


def calculate_level_and_Trend_and_seasonality(dataframe, initial_level, initial_trend, seasonalityData, demand_Column_Name, Sequence_Column_Name, Average_Seasonal_Factors_Values, A_Smoothing_Constant = 0.5, B_Smoothing_Constant = 0.5, G_Smoothing_Constant = 0.5):
    level_list = []
    level_values = {}
    trend_list = []
    trend_values = {}
    seasonality_list = []
    seasonality_values = {}
    forecast_values = {}
    seaonsal_factor_values_sequence = {}
    for x in seasonalityData.keys():
        level_values.update({x: []})
        trend_values.update({x:[]})
        seasonality_values.update({x:[]})
        seaonsal_factor_values_sequence.update({x:{"Level":0, "Trend":0}})
    count = 0
    if count == 0:
        previous_level = initial_level
        previous_Trend = initial_trend
    Sequence_Column = dataframe[Sequence_Column_Name]
    arrive_time_column = dataframe[demand_Column_Name]
    for sequence, arrive_time in zip(Sequence_Column, arrive_time_column):
        seasonalFactor = Average_Seasonal_Factors_Values.get(sequence)
        Level = A_Smoothing_Constant * (arrive_time/seasonalFactor[0]) + (1-A_Smoothing_Constant) * (previous_level + previous_Trend)
        Trend = B_Smoothing_Constant * (Level - previous_level) + (1-B_Smoothing_Constant) * previous_Trend
        Seasonality = G_Smoothing_Constant * (arrive_time/Level) + (1-G_Smoothing_Constant)*seasonalFactor[0]
        Forecast = (Level + Trend) * Seasonality
        forecast_values.update({arrive_time:Forecast})
        level_list.append(Level)
        trend_list.append(Trend)
        level_values.get(sequence).append(Level)
        trend_values.get(sequence).append(Trend)
        seasonality_list.append(Seasonality)
        seasonality_values.get(sequence).append(Seasonality)
        previous_level = Level
        previous_Trend = Trend
        seaonsal_factor_values_sequence.get(sequence).update({"Level":Level})
        seaonsal_factor_values_sequence.get(sequence).update({"Trend":Trend})
    return level_list, level_values, trend_list, trend_values, seasonality_list, seasonality_values, forecast_values, Level, Trend, Seasonality, seaonsal_factor_values_sequence


def forecat_error(forecast_values):
    forecasted_error_list = []
    for x in forecast_values.keys():
        Forecasted_error = forecast_values.get(x) - x
        forecasted_error_list.append(Forecasted_error)
    return forecasted_error_list


def Mape_error(forecasted_error_list):
    import numpy
    Mape_error_list = []
    count = 1
    while (count < len(forecasted_error_list)):
        mean = numpy.mean(forecasted_error_list[0:count])
        Mape_error_list.append(mean)
        count += 1
    return Mape_error_list


def bias_error(forecasted_error_list):
    Bias_error_list = []
    count = 1
    while (count < len(forecasted_error_list)):
        sum_value = sum(forecasted_error_list[0:count])
        Bias_error_list.append(sum_value)
        count += 1
    return Bias_error_list


def tracking_signal(forecasted_error_list, Bias_error_list, periodlist):
    Absolute_Derivation_list = [abs(x) for x in forecasted_error_list]
    count = 1
    start = 0
    Tracking_Signal_list = []
    while (count < len(Absolute_Derivation_list)):
        Mad_value = sum(Absolute_Derivation_list[0:count]) / periodlist[start]
        Tracking_Signal = Bias_error_list[start] / Mad_value
        Tracking_Signal_list.append(Tracking_Signal)
        start += 1
        count += 1
    return Tracking_Signal_list


def Calculate_Best_Smoothing_Constant(dataframe, initial_level, initial_trend, seasonalityData, demand_Column_Name, Sequence_Column_Name, Average_Seasonal_Factors_Values, periodlist):
    from itertools import product

    # Define the set of values
    values = [0, 0.25, 0.5, 0.75, 1]
    smoothing_constnat = ""
    # Generate all permutations of 4 values
    permutations = list(product(values, repeat=3))

    permutation_dicts = []
    for perm in permutations:
        a_value, b_value, c_value = perm
        permutation_dict = {"A": a_value, "B": b_value, "C": c_value}
        permutation_dicts.append(permutation_dict)
    import numpy
    bigvaluelist = {}
    for x in permutation_dicts:
        values_acheived = []
        level_list, level_values, trend_list, trend_values, seasonality_list, seasonality_values, forecast_values, Level, Trend, seasonality, seaonsal_factor_values_sequence = calculate_level_and_Trend_and_seasonality(dataframe, initial_level, initial_trend, seasonalityData, demand_Column_Name, Sequence_Column_Name, Average_Seasonal_Factors_Values, x.get("A"), x.get("B"), x.get("C"))
        forecasted_error_list = forecat_error(forecast_values)
        Mape_error_list = Mape_error(forecasted_error_list)
        bias_error_list = bias_error(forecasted_error_list)
        tracking_signal_list = tracking_signal(forecasted_error_list, bias_error_list, periodlist)
        Mape_Mean = numpy.mean(Mape_error_list)
        bias_Mean = numpy.mean(bias_error_list)
        tracking_signal_Mean = numpy.mean(tracking_signal_list)
        values_acheived.append(Mape_Mean)
        values_acheived.append(bias_Mean)
        values_acheived.append(tracking_signal_Mean)
        bigvaluelist.update({str(x):values_acheived})
    best_smoothing_constant_list = []
    best_smoothing_constant_value = {}
    for x in bigvaluelist.keys():
        mean_error = numpy.mean(bigvaluelist.get(x))
        best_smoothing_constant_list.append(abs(mean_error))
        best_smoothing_constant_value.update({x:abs(mean_error)})
    best_smoothing_constant_list_sorted = sorted(best_smoothing_constant_list)
    for key, value in best_smoothing_constant_value.items():
        if value == best_smoothing_constant_list_sorted[0]:
            smoothing_constnat = key
    return smoothing_constnat


def winters_method_forecast(dataframe, initial_level, initial_trend, seasonalityData, demand_Column_Name, Sequence_Column_Name, smoothing_constnat, NumberOfPeriods, Average_Seasonal_Factors_Values):
    # Use the following euqation =(S3+T3)*U3
    # S Column = Level Column 
    # T Column = Trend Column
    # U Column = Seasonality Column 
    import ast
    converted_dict = ast.literal_eval(smoothing_constnat)

    level_list, level_values, trend_list, trend_values, seasonality_list, seasonality_values, forecast_values, Level, Trend, seasonality, seaonsal_factor_values_sequence = calculate_level_and_Trend_and_seasonality(dataframe, initial_level, initial_trend, seasonalityData, demand_Column_Name, Sequence_Column_Name, Average_Seasonal_Factors_Values, converted_dict.get("A"), converted_dict.get("B"), converted_dict.get("C"))

    period_number_list = [x for x in range(1,NumberOfPeriods+1)]
    Sequence_Column = dataframe[Sequence_Column_Name]
    Sequence = set(Sequence_Column.to_list())
    forecast_timesequnce_values = {}
    for x in range(len(period_number_list)):
        Forecated_value = (seaonsal_factor_values_sequence.get(list(Sequence)[x]).get("Level") + period_number_list[x] * seaonsal_factor_values_sequence.get(list(Sequence)[x]).get("Trend")) * Average_Seasonal_Factors_Values.get(list(Sequence)[x])[0]
        forecast_timesequnce_values.update({list(Sequence)[x]:Forecated_value})
    return forecast_timesequnce_values


def forecast_1_period(dataframe, Sequence_Column_Name, demand_Column_Name):
    import numpy
    mean_demand_column = numpy.mean(dataframe[demand_Column_Name])
    Sequence_Column = dataframe[Sequence_Column_Name]
    Sequence = set(Sequence_Column.to_list())
    forecast_timesequnce_values = {list(Sequence)[0]:mean_demand_column}
    return forecast_timesequnce_values


def winters_method(dataframe, Sequence_Column_Name, Fleet_Column_Name, demand_Column_Name, output_Filled_CSV):
    print("---- Starting Winter's Method Calculation ----")
    print("---- Using the %s Column As the Seasonality Factors ----" %(Sequence_Column_Name))
    seasonalityData = Column_Seasonality(dataframe, Sequence_Column_Name)
    print("---- Finished Organizing the Similiar Seasonality Factors ----")
    print(seasonalityData)
    print("---- Calculating the Number of Periods ----")
    NumberOfPeriods = Calculate_Number_Of_Periods(seasonalityData)
    print("---- Finished Calculated Number of Periods: %s"%(str(NumberOfPeriods)))
    if NumberOfPeriods > 1: 
        print("---- Starting Filling Columns to Ensure Seasonality is Constant -----")
        filtered_training_data = Fill_Columns(dataframe, Sequence_Column_Name, seasonalityData, Fleet_Column_Name, output_Filled_CSV)
        print("---- Finished Created the filled out columns with copies outputed to csv to showcase which ones were added path: %s"%(output_Filled_CSV))
        print(filtered_training_data.head())
        print("---- Determining Seasonalized Demand Data ----")
        seasonalized_Demand_list, Sesonalized_Demand_values = winters_method_Seasonalized_Demand(filtered_training_data, NumberOfPeriods, Sequence_Column_Name, demand_Column_Name, seasonalityData)
        print("---- Finished Finding the Seasonalized Demand Data ----")
        #print(seasonalized_Demand_list)
        #print(Sesonalized_Demand_values)
        print("---- Determining De-Seasonalized Demand Data ----")
        De_seasonalized_Demand_list, De_Sesonalized_Demand_values, periodlist = winters_method_De_seasonalized_Demand(filtered_training_data, NumberOfPeriods, seasonalityData, seasonalized_Demand_list, Sesonalized_Demand_values)
        print("---- Finished Finding the De-Seasonalized Demand Data ----")
        print("---- Determining Seasonality of Demand Data ----")
        Seasonal_Factors_List, Seasonal_Factors_Values = winters_method_Seasonality(filtered_training_data, demand_Column_Name, seasonalityData, De_seasonalized_Demand_list, De_Sesonalized_Demand_values)
        print("---- Finished Finding the Seasonality of Demand Data ----")
        print("---- Determining the Average Seasonality per Seasonablity Factor ----")
        Average_Seasonal_Factors_Values = winters_method_average_Seasonality(seasonalityData, Seasonal_Factors_Values)
        print("---- Finished Finding the Average Seasonality per Seasonality Factor ----")
        print("---- Finding the Intial Level and Intial Trend ----")
        initial_level, initial_trend = Initial_Level_And_Trend(De_seasonalized_Demand_list, periodlist)
        print("---- Finish Finding the Initial Level: %s and Initial Trend: %s" %(str(initial_level), str(initial_trend)))
        print("---- Determining the Best Smoothing Constant for the datframe ----")
        smoothing_constnat = Calculate_Best_Smoothing_Constant(filtered_training_data, initial_level, initial_trend, seasonalityData, demand_Column_Name, Sequence_Column_Name, Average_Seasonal_Factors_Values, periodlist)
        print("---- Finished Finding the best smoothing constnat is: %s"%(smoothing_constnat))
        print("---- Starting forecasting values of future values using all the smoothing consant calcualted ----")
        forecast_timesequnce_values = winters_method_forecast(filtered_training_data, initial_level, initial_trend, seasonalityData, demand_Column_Name, Sequence_Column_Name, smoothing_constnat, NumberOfPeriods, Average_Seasonal_Factors_Values)
        print("---- Finished forecasting all Seasonality factors ----")
        #print(forecast_timesequnce_values)
    else:
        forecast_timesequnce_values = forecast_1_period(dataframe, Sequence_Column_Name, demand_Column_Name)
    return forecast_timesequnce_values


