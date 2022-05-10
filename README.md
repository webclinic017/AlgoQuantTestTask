# AlgoQuantTestTask
Entrance task from the AlgoQuant company for Mikhail Pravdukhin.

### Description
The repository implements backtesting of the Long/Short trading strategy using the binance spot market data. The idea of the strategy is to buy top of the most unprofitable instruments and sell top of the most profitable ones in relation to the market. For more information about the logic of the trading strategy, see LongShortTradingStrategy.py.

### Navigation
**LongShortTradingStrategy.py** - class that implements the logic of the strategy.  
**tools.py** - auxiliary functions for loading the data, running the optimization of the strategy parameters, visualizing the results of the trading.  
**OneHourDataResults.ipynb** - presentation of backtesting for 1hour data.  
**data** - time bars datasets for various binance spot market instruments (1hour and daily bars are provied).

### Requirements
Following python 3 libraries are required:
* pandas
* numpy
* matplotlib
* optuna
* sklearn

