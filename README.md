# Predicting Room Occupancy Using Machine Learning Models

## Data Set 
This dataset was obtained via "Data Science Dojo" at https://code.datasciencedojo.com/datasciencedojo/datasets/tree/master/Occupancy%20Detection. It is listed as originating from a publication at https://www.researchgate.net/profile/Luis_Candanedo_Ibarra/publication/285627413_Accurate_occupancy_detection_of_an_office_room_from_light_temperature_humidity_and_CO2_measurements_using_statistical_learning_models/links/5b1d843ea6fdcca67b690c28/Accurate-occupancy-detection-of-an-office-room-from-light-temperature-humidity-and-CO2-measurements-using-statistical-learning-models.pdf?origin=publication_detail   

The data is provided in three parts (`datatest.csv`, `datatest2.csv`, `data training.csv`), which can be recombined to form the full data set. The combined dataset has 20560 rows and 7 columns. It includes one categoriacl variable, `Occupancy` which is the target variable for predictive models.

### Data Dictionary 
The original data set contains the following variables
| Atrribute Name 	| Description                                                                                           | Data Type    	| 
|----------------	|------------------------------------------------------------------------------------------------------	|--------------	|
| Date           	| Date and time of data collection (MM/DD/YYYY HH:MM)                                                  	| Object  	|
| Temperature    	| Temperature in Celcius                                                                              	| float64 	| 
| Humidity       	| Relative humidity (percent)                                                                         	| float64 	|
| Light          	| Measure of light in Lux                                                                             	| float64 	|
| CO2            	| CO2 measured in parts per million (ppm)                                                              	| float64 	|
| HumidityRatio  	| Measure of kg of water vapor / kg air in room                                                        	| float64 	|
| Occupancy      	| 1 if a room is occupied, 0 if a room is unoccupied                                                   	| int64 	|

## Analysis

### File Structure
***ADD FILE STRUCTURE HERE AND HOW THEY ARE CONNECTED/WHAT TO RUN***
