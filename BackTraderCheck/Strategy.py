import backtrader as bt
from operator import itemgetter
from numpy import isnan, ceil


class Strategy(bt.Strategy):

    def __init__(self):
        self.__rets = {ticker: bt.ind.PctChange(self.getdatabyname(ticker), period=1) for ticker in self.getdatanames()}
        self.__open_price = {ticker: None for ticker in self.getdatanames()}

        # Helper functions to filter returns list, pair = (ticker, return)
        self.__nan_filter_func = lambda pair: not isnan(pair[-1])
        self.__pos_filter_func = lambda pair: self.__open_price[pair[0]] is None
        self.__filter_func = lambda pair: self.__nan_filter_func(pair) and self.__pos_filter_func(pair)
        self.__up_filter = lambda pair: pair[-1] > 0
        self.__down_filter = lambda pair: pair[-1] < 0
        self.__sort_func = itemgetter(-1)

        # We have to track current cash ourselves because backtrader updates it only after the 'next' method is over
        # while we need to know how much money we have in the middle of the 'next' method before we open new positions
        self.__cash = self.broker.cash

        # We also need to track how much we shorted to take it into account
        # because when we open new positions we don't want to have leverage > 1
        self.__cash_shorted_abs = 0

        self.down_q = 0.07
        self.up_q = 0.07

    def next(self):
        # First, close all opened position
        self.__close_positions()
        # Then open new positions
        self.__open_positions()
        print(self.datas[0].datetime.datetime(0), self.broker.getvalue())

    def __close_positions(self):
        for ticker in self.getdatanames():
            pos = self.getpositionbyname(ticker)
            data = self.getdatabyname(ticker)
            # If there is no price, we can't close position. Hence, it will be close further
            if pos and (not isnan(data.close[0])):
                self.__cash += pos.size * data.close[0]
                if pos.size < 0:
                    # If short position is closed, then we update the corresponding variable
                    self.__cash_shorted_abs += pos.size * self.__open_price[ticker]
                self.__open_price[ticker] = None
                self.close(data, pos.size, price=data.close[0])

    def __open_positions(self):
        # Take into account shorted cash
        # Once we subtract because the shorted cache was added before
        # The second time - for taking into account that it is no longer possible to open deals for this amount of cash
        cash_available = self.__cash - 2 * self.__cash_shorted_abs
        if abs(cash_available) <= 1e-6:
            return

        # filter instruments with empty current return and for which positions are already opened
        ticker_ret_pairs = [(ticker, self.__rets[ticker][0]) for ticker in self.getdatanames()]
        ticker_ret_pairs = list(filter(self.__filter_func, ticker_ret_pairs))

        # short position are selected only from instruments with negative return
        up_ticker_ret_pairs = sorted(list(filter(self.__up_filter, ticker_ret_pairs)),
                                     key=self.__sort_func, reverse=True)

        # long position are selected only from instruments with positive return
        down_ticker_ret_pairs = sorted(list(filter(self.__down_filter, ticker_ret_pairs)),
                                       key=self.__sort_func)

        # number of instruments to buy/sell is calculated based on their total number
        total_number_of_instruments = len(ticker_ret_pairs)
        n = int(ceil(total_number_of_instruments * self.down_q))
        long_tickers = self.__get_top_tickers(down_ticker_ret_pairs, n)
        n = int(ceil(total_number_of_instruments * self.up_q))
        short_tickers = self.__get_top_tickers(up_ticker_ret_pairs, n)

        n_tickers_long = len(long_tickers)
        n_tickers_short = len(short_tickers)
        n_tickers = n_tickers_long + n_tickers_short
        if n_tickers != 0:
            cap_per_ticker = cash_available / n_tickers

            for ticker in long_tickers:
                data = self.getdatabyname(ticker)
                price = data.close[0]
                size = int(cap_per_ticker / price)
                self.__open_price[ticker] = price
                # cash is reduced when long position is opened
                self.__cash -= price * size
                self.buy(data, size, data.close[0])

            for ticker in short_tickers:
                data = self.getdatabyname(ticker)
                price = data.close[0]
                size = int(cap_per_ticker / price)
                self.__open_price[ticker] = price
                money = price * size
                # cash is increased when short position is opened
                self.__cash += money
                # variable that stores shorted amount is updated too
                self.__cash_shorted_abs += money
                self.sell(data, size, data.close[0])

    @staticmethod
    def __get_top_tickers(pairs, n):
        return [pair[0] for pair in pairs[:n]]


