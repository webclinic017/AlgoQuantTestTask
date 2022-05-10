from LongShortTradingStrategy import LongShortTradingStrategy
from tools import *


bars_data = load_ohlc_data('data/binance_1h')
prices_data = get_bar_value_for_all_tickers(bars_data, 'close')
strategy = LongShortTradingStrategy(prices_data, benchmark_ticker='BTCUSDT')
strategy.set_strategy_params(up_q=0.07, down_q=0.07, max_t_in_pos=1)
strategy.run()
123