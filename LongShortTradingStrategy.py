import numpy as np
from pandas import DataFrame, Series
from operator import itemgetter


class LongShortTradingStrategy:
    """
    Implements logic of the simple trading strategy based on the following idea: if the current return of an instrument
    strongly outperforms the market, it is expected that the instrument's price will go down in the future;
    at the same time if the return underperforms the market, it is expected that the price will go up.

    Hence, at each time period, the distribution of returns is calculated, and the upper quantile is sold,
    the lower quantile is bought.

    The position is closed when at least one of the following conditions is satisfied:
        1. Number of time periods in position exceeds the given limit (max_t_in_pos).
        2. Current position return beyond the given limits ([max_profit, -max_loss]).
        3. Number of consecutive time periods when position is profitable/unprofitable exceed the give limit
        (max_t_in_profit/loss).

    Closing condition depends on the strategy parameters that can be set in set_strategy_params(...).

    To simulate the trading strategy on the given data use run()
    """

    class TickerInfo:
        def __init__(self, pos=0, buy_price=None, sell_price=None, time=0, profit_loss_counter=0, is_profit=False):
            self.pos = pos
            self.buy_price = buy_price
            self.sell_price = sell_price
            self.time = time
            self.profit_loss_counter = profit_loss_counter
            self.is_profit = is_profit

        def clear(self):
            self.__init__()

    def __init__(self, prices_data, start_cap=1000000.0, benchmark_ticker=None):
        """
        Parameters
        ----------
        prices_data : DataFrame
            close prices for given instruments (columns) and time period(index)
        start_cap : float
            available money at the beginning of the strategy
        benchmark_ticker : str or None
            instrument that is used to calculate strategy returns correlation
            if None, correlation is not calculated
        """
        self.__prices_data = prices_data
        self.__returns_data = prices_data.values[1:] / prices_data.values[:-1] - 1
        self.__returns_data = DataFrame(self.__returns_data, index=prices_data.index[1:], columns=prices_data.columns)

        self.__time_periods_in_one_year = self.__prices_data.index[1] - self.__prices_data.index[0]
        self.__time_periods_in_one_year = self.__time_periods_in_one_year.total_seconds()
        # Binance trading year has 365 days
        self.__time_periods_in_one_year = 365.0 * 24 * 3600 / self.__time_periods_in_one_year
        self.__sqrt_time_periods_in_one_year = np.sqrt(self.__time_periods_in_one_year)

        self.start_cap = start_cap
        self.benchmark_ticker = None if benchmark_ticker not in prices_data.columns else benchmark_ticker

        # Set default params
        self.set_strategy_params()

        # Create cashes for running strategy
        self.__clear_work_variables()

        # Helper functions to filter returns list, pair = (ticker, return)
        self.__nan_filter_func = lambda pair: not np.isnan(pair[-1])
        self.__pos_filter_func = lambda pair: not self.__tickers_info[pair[0]].pos
        self.__filter_func = lambda pair: self.__nan_filter_func(pair) and self.__pos_filter_func(pair)
        self.__up_filter = lambda pair: pair[-1] > 0
        self.__down_filter = lambda pair: pair[-1] < 0
        self.__sort_func = itemgetter(-1)

    def set_strategy_params(self, up_q=0.05, down_q=0.05, max_profit=0.05, max_loss=0.03,
                            max_t_in_pos=10, max_t_in_profit=None, max_t_in_loss=None):
        """
        Sets strategy parameters that are used in simulation of trading

        Parameters
        ----------
        up_q : [0, 1]
            upper quantile of returns distribution that is used to open long positions
        down_q : [0, 1]
            lower quantile of returns distributions that is used to open short positions
        max_profit : > 0
            position is closed, if position return >= max_profit
        max_loss : > 0
            position is closed, if position return <= -max_loss
        max_t_in_pos : int > 0
            position is closed, if number of time periods in position >= max_t_pos
        max_t_in_profit : int > 0 or None
            position is closed, if number of consecutive profitable time periods >= max_t_in_profit
            if None, this check is disabled
        max_t_in_loss :  int > 0 or None
            position is closed, if number of consecutive unprofitable time periods >= max_t_in_loss
            if None, this check is disabled

        Returns
        -------
        None
        """
        self.up_q = up_q
        self.down_q = down_q
        self.max_profit = max_profit
        self.max_loss = max_loss
        self.max_t_in_pos = max_t_in_pos
        self.max_t_in_profit = max_t_in_profit
        self.max_t_in_loss = max_t_in_loss

    def run(self, log_trades=False):
        """
        Simulates trading using the rules of the strategy.

        Trading simulation is terminated if strategy capital becomes <= 0.0.
        Such a situation means that a sequence of unprofitable short trades took a place.

        Parameters
        -------
        log_trades : bool
            if True, all trades are saved to self.trades_logs

        Returns
        -------
        None

        See Also
        -------
        After simulation is done, the following values are calculated:
        self.cap : Series
            values of the strategy capital for given in self.prices_data time periods
        self.cap_ret : Series
            strategy returns for given in self.prices_data time periods
        self.avg_return : float
            average strategy return
        self.volatility : float
            standard deviation of strategy returns
        self.sharpe : float
            Sharpe ratio of the strategy
        self.cagr : float
            compound annual growth rate
        self.correlation :
            correlation coefficient of benchmark returns and strategy returns
        """
        self.__clear_work_variables()

        if log_trades:
            self.trades_logs = []

        for t in self.__returns_data.index:
            self.__last_known_prices = self.__prices_data.loc[t].fillna(self.__last_known_prices)

            self.__close_positions_if_needed(t)
            self.__open_positions_if_needed(t)

            self.__cap.append(self.__calculate_cap(t, self.__current_portf))

            if self.__cap[-1] <= 0.0:
                break

        self.__cap = Series(self.__cap, self.__prices_data.index[1:len(self.__cap) + 1], name='Strategy capital')
        self.__cap_ret = Series((self.cap[1:].values / self.cap[:-1].values) - 1, self.__cap.index[1:],
                                name='Strategy returns')

    @property
    def prices_data(self):
        return self.__prices_data

    @property
    def returns_data(self):
        return self.__returns_data

    @property
    def cap(self):
        """
        Values of the strategy capital for given in self.prices_data time periods
        """
        return self.__cap

    @property
    def cap_ret(self):
        """
        Strategy returns for given in self.prices_data time periods
        """
        return self.__cap_ret

    @property
    def avg_return(self):
        """
        Average strategy return
        """
        self.__calculate_avg_return()
        return self.__avg_return

    def __calculate_avg_return(self):
        if (self.__avg_return is not None) or (self.__cap_ret is None):
            return

        self.__avg_return = self.__cap_ret.mean()

    @property
    def volatility(self):
        """
        Standard deviation of strategy returns
        """
        self.__calculate_volatility()
        return self.__volatility

    def __calculate_volatility(self):
        if (self.__volatility is not None) or (self.__cap_ret is None):
            return

        self.__volatility = self.__cap_ret.std()

    @property
    def sharpe(self):
        """
        Sharpe ratio of the strategy
        """
        self.__calculate_sharpe()
        return self.__sharpe

    def __calculate_sharpe(self):
        if (self.__sharpe is not None) or (self.__cap_ret is None):
            return

        self.__sharpe = self.__sqrt_time_periods_in_one_year * self.avg_return / self.volatility

    @property
    def cagr(self):
        """
        Compound annual growth rate
        """
        self.__calculate_cagr()
        return self.__cagr

    def __calculate_cagr(self):
        if (self.__cagr is not None) or (self.__cap_ret is None):
            return

        ratio = self.__cap.values[-1] / self.cap.values[0]

        self.__cagr = np.nan if ratio <= 0.0 else np.power(ratio, self.__time_periods_in_one_year / self.cap.size) - 1

    @property
    def correlation(self):
        """
        Correlation coefficient of benchmark returns and strategy returns
        """
        self.__calculate_correlation()
        return self.__correlation

    def __calculate_correlation(self):
        if (self.__correlation is not None) or (self.__cap_ret is None) or (self.benchmark_ticker is None):
            return

        self.__correlation = np.corrcoef(self.__cap_ret.values,
                                         self.__returns_data[self.benchmark_ticker].values[1:])[1, 0]

    def __clear_work_variables(self):
        self.__cap = []
        self.__cap_ret = None

        self.__avg_return = None
        self.__volatility = None
        self.__sharpe = None
        self.__cagr = None
        self.__correlation = None

        self.__tickers_info = {ticker: self.TickerInfo() for ticker in self.__returns_data.columns}
        self.__current_portf = dict()  # {ticker : quantity}
        self.__cash = self.start_cap

        self.__orders = dict()  # {ticker : quantity}
        self.__can_make_trades = True

        self.__last_known_prices = self.__prices_data.iloc[0]

        self.trades_logs = None

    def __calculate_portfolio_value(self, t, portf):
        return (self.__last_known_prices[portf.keys()].values * [item[-1] for item in portf.items()]).sum()

    def __calculate_cap(self, t, portf):
        pos_value = self.__calculate_portfolio_value(t, portf)
        return pos_value + self.__cash

    def __make_trades(self, t, orders):
        for ticker in orders:
            if self.__current_portf.get(ticker) is None:
                self.__current_portf[ticker] = orders[ticker]
            else:
                self.__current_portf[ticker] += orders[ticker]

            if abs(self.__current_portf[ticker]) <= 1e-6:
                self.__current_portf.pop(ticker)

            if self.trades_logs is not None:
                self.trades_logs.append((t, ticker, orders[ticker], self.__last_known_prices[ticker]))

        self.__cash -= self.__calculate_portfolio_value(t, orders)

    @staticmethod
    def __get_top_tickers(pairs, n):
        return [pair[0] for pair in pairs[:n]]

    def __calculate_position_ret(self, t, ticker, info):
        if info.pos == 1:
            return self.__prices_data.loc[t, ticker] / info.buy_price - 1
        elif info.pos == -1:
            return info.sell_price / self.__prices_data.loc[t, ticker] - 1

    def __close_positions_if_needed(self, t):
        for ticker in self.__current_portf:
            info = self.__tickers_info[ticker]
            info.time += 1

            close_pos = info.time >= self.max_t_in_pos

            if (self.max_t_in_profit is not None) and (self.max_t_in_loss is not None):
                last_known_price = self.__last_known_prices[ticker]
                is_profit = (info.pos == 1) and (last_known_price > info.buy_price)
                is_profit |= (info.pos == -1) and (last_known_price < info.sell_price)

                if info.is_profit == is_profit:
                    info.profit_loss_counter += 1
                else:
                    info.is_profit = is_profit
                    info.profit_loss_counter = 1

                limit = self.max_t_in_profit if is_profit else self.max_t_in_loss
                close_pos |= info.profit_loss_counter >= limit

            # First, counters are updated, then current price is checked
            if np.isnan(self.__prices_data.loc[t, ticker]):
                # Position can't be closed if there's no price
                continue

            # There is no need to calculate current ret if we already decided to close position
            if not close_pos:
                ret = self.__calculate_position_ret(t, ticker, info)
                close_pos = (ret >= self.max_profit) or (ret <= -self.max_loss)

            if close_pos:
                self.__orders.update({ticker: -self.__current_portf[ticker]})
                info.clear()

        if not (len(self.__orders) == 0):
            self.__make_trades(t, self.__orders)
            self.__orders.clear()
            self.__can_make_trades = self.__cash > 0.0

    def __open_positions_if_needed(self, t):
        if not self.__can_make_trades:
            return

        ticker_ret_pairs = [(ticker, self.__returns_data.loc[t, ticker]) for ticker in self.__returns_data.columns]
        # filter instruments with empty current return and for which positions are already opened
        ticker_ret_pairs = list(filter(self.__filter_func, ticker_ret_pairs))

        # short position are selected only from instruments with negative return
        up_ticker_ret_pairs = sorted(list(filter(self.__up_filter, ticker_ret_pairs)),
                                     key=self.__sort_func, reverse=True)

        # long position are selected only from instruments with positive return
        down_ticker_ret_pairs = sorted(list(filter(self.__down_filter, ticker_ret_pairs)),
                                       key=self.__sort_func)

        # number of instruments to buy/sell is calculated based on their total number
        total_number_of_instruments = len(ticker_ret_pairs)
        n = int(np.ceil(total_number_of_instruments * self.down_q))
        long_tickers = self.__get_top_tickers(down_ticker_ret_pairs, n)
        n = int(np.ceil(total_number_of_instruments * self.up_q))
        short_tickers = self.__get_top_tickers(up_ticker_ret_pairs, n)

        n_tickers = len(long_tickers) + len(short_tickers)
        if n_tickers != 0:
            cap_per_ticker = self.__cash / n_tickers

            for ticker in long_tickers:
                price = self.__prices_data.loc[t, ticker]
                self.__orders.update({ticker: cap_per_ticker / price})
                self.__tickers_info[ticker].pos = 1
                self.__tickers_info[ticker].buy_price = price

            for ticker in short_tickers:
                price = self.__prices_data.loc[t, ticker]
                self.__orders.update({ticker: -cap_per_ticker / price})
                self.__tickers_info[ticker].pos = -1
                self.__tickers_info[ticker].sell_price = price

            self.__make_trades(t, self.__orders)
            self.__orders.clear()
            self.__can_make_trades = False
