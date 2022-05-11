# AlgoQuantTestTask
Entrance task from the AlgoQuant company for Mikhail Pravdukhin.

### Description
The repository implements backtesting of the Long/Short trading strategy using the binance spot market data. The idea of the strategy is to buy top of the most unprofitable instruments and sell top of the most profitable ones in relation to the market. For more information about the logic of the trading strategy, see LongShortTradingStrategy.py.

### Navigation
**OneHourDataResults.ipynb** - presentation of backtesting for 1hour data.  
**DailyDataResults.ipynb** - presentation of backtesting for daily data.  
**LongShortTradingStrategy.py** - class that implements the logic of the strategy.  
**tools.py** - auxiliary functions for loading the data, running the optimization of the strategy parameters, visualizing the results of the trading.  
**data** - time bars datasets for various binance spot market instruments (1hour and daily bars are provied).

### How to view the project
Fisrt, look at OneHourDataResults.ipynb - it presents the complete history of the research performed for the 1hor data.  
While viewing the OneHourDataResults.ipynb, you can open LongShortTradingStrategy.py and get to know with the details of the trading strategy.  
If you are interested in the logic of the strategy parameters optimization, look at the optimize_strategy_params(...) function in tools.py.  

Once you've done with OneHourDataResults.ipynb, you can look at DailyDataResults.ipynb. The presentation for the daily data is not so detailed because the daily dataset is much poorer that 1h. But it is still interesting to see how strategy performs on daily data.

### Requirements
Following python 3 libraries are required:
* pandas
* numpy
* matplotlib
* optuna
* sklearn

