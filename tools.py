from LongShortTradingStrategy import LongShortTradingStrategy
from numpy import isnan, mean, power
import pandas as pd
import matplotlib.pyplot as plt
import optuna
import os


def load_ohlc_data(folder_path):
    bars_data = []
    for file_name in os.listdir(folder_path):
        ticker = file_name[:file_name.index('.')]
        df = pd.read_csv(os.path.join(folder_path, file_name),
                         index_col='timestamp', parse_dates=[0])
        columns = pd.MultiIndex.from_tuples([(ticker, column) for column in df.columns])
        df.columns = columns
        bars_data.append(df)
    return pd.concat(bars_data, axis=1)


def get_bar_value_for_all_tickers(bars_data, value):
    tickers = bars_data.columns.get_level_values(0).unique()
    close_columns = [(ticker, value) for ticker in tickers]
    values_data = bars_data[close_columns]
    values_data.columns = tickers
    return values_data


def optimize_strategy_params(train_prices_data, up_q_bounds, down_q_bounds, max_profit_bounds, max_loss_bounds,
                             max_t_in_pos_bounds, optimize_max_t_in_profit_and_loss=False,
                             n_trials=100):
    strategy = LongShortTradingStrategy(train_prices_data, benchmark_ticker='BTCUSDT')

    def objective(trial):
        up_q = trial.suggest_float('up_q', *up_q_bounds)
        down_q = trial.suggest_float('down_q', *down_q_bounds)
        max_profit = trial.suggest_float('max_profit', *max_profit_bounds)
        max_loss = trial.suggest_float('max_loss', *max_loss_bounds)
        max_t_in_pos = trial.suggest_int('max_t_in_pos', *max_t_in_pos_bounds)
        max_t_in_profit = None
        max_t_in_loss = None
        if optimize_max_t_in_profit_and_loss:
            # There is no sense in max_t_in_profit/loss > max_t_in_pos
            max_t_in_profit = trial.suggest_int('max_t_in_profit', 1, max_t_in_pos)
            max_t_in_loss = trial.suggest_int('max_t_in_loss', 1, max_t_in_pos)

        strategy.set_strategy_params(up_q, down_q, max_profit, max_loss, max_t_in_pos, max_t_in_profit, max_t_in_loss)
        strategy.run()
        if isnan(strategy.cagr):
            # If CAGR is nan, the final value of strategy capital < 0.
            # Such a result is a trash, hence we penalize it
            return -100.0
        else:
            return strategy.sharpe

    # Set default sampler with seed to make optimization results reproducible
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    strategy.set_strategy_params(**study.best_params)
    strategy.run()
    return strategy, study.best_params


def plot_strategy_results(strategy, name):
    plt.figure(figsize=(20, 10))
    pd.Series((strategy.cap * 100 / strategy.cap.values[0]) - 100, strategy.cap.index,
              name='Strategy cumulative return').plot()
    benchmark_prices = strategy.prices_data[strategy.benchmark_ticker].values[1:]
    pd.Series((benchmark_prices * 100 / benchmark_prices[0]) - 100, strategy.cap.index,
              name=strategy.benchmark_ticker + ' cumulative return').plot()
    plt.xlabel('')
    plt.ylabel('Cumulative return, %', size=20)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.legend(fontsize=15)
    tmp = (round(strategy.sharpe, 2), str(round(strategy.cagr * 100, 2)) + '%',
           str(round((power(strategy.cagr + 1, 1/365) - 1) * 100, 2)) + '%',
           str(round(mean(strategy.trade_returns) * 100, 2)) + '%',
           strategy.benchmark_ticker, round(strategy.correlation, 2))
    plt.title(name + '\nSharpe ratio = %s\nCAGR = %s\nAverage daily return = %s'
                     '\nAverage return per trade = %s\nCorrelation with %s = %s' % tmp,
              fontsize=20)
    plt.show()


