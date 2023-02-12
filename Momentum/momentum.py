import numpy as np


def zscore_strategy(data_prices, window=5, threshold=0.0, long_only=False):
    assert len(data_prices.columns)==1
    data = data_prices.copy()

    # Calculate the log returns of the data
    data['returns'] = np.log(data['Close'] / data['Close'].shift(1))
    # Calculate the rolling average of the returns
    data['average'] = data['returns'].rolling(window=window).mean()
    
    # Calculate the standard deviation of the returns
    data['std'] = data['returns'].rolling(window=window).std()
    
    # Calculate the z-score of the returns
    data['z_score'] = (data['returns'] - data['average']) / data['std']
    
    # Create a buy/sell signal when the z-score is greater than the threshold
    data['signal'] = (np.where(data['z_score'] > threshold, 1, 0) +
                      (not long_only)*(np.where(data['z_score'] < -threshold, -1, 0)))
                          
    # Calculate the returns for each trade
    data['trade_returns'] = data['signal'].shift(1) * (data['Close']/data['Close'].shift(1)-1)
    
    # Calculate the cumulative returns for all trades
    data['cumulative_returns'] = data['trade_returns'].cumsum()

    return data

def sign_strategy(data_prices, window=252, long_only=False):
    assert len(data_prices.columns)==1
    data = data_prices.copy()

    # Calculate the log returns of the data
    data['sign'] = np.sign((data['Close'] / data['Close'].shift(window))-1)

    # Create a buy/sell signal when the z-score is greater than the threshold
    data['signal'] = (np.where(data['sign'] > 0, 1, 0) +
                      (not long_only)*(np.where(data['sign'] < 0, -1, 0)))
                          
    # Calculate the returns for each trade
    data['trade_returns'] = data['signal'].shift(1) * (data['Close']/data['Close'].shift(1)-1)
    
    # Calculate the cumulative returns for all trades
    data['cumulative_returns'] = data['trade_returns'].cumsum()

    return data

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Get data 
    from getdata import FinancialDataProvider
    dataprovider = FinancialDataProvider()

    # Set the ticker we want to trade
    ticker = 'SPY'

    # Set the dates
    start_date = '2000-01-02'
    end_date = '2020-02-08'

    # load the data
    data_provider = FinancialDataProvider()
    all_data = data_provider.get_ticker(ticker, start_date, end_date)

    long_only = zscore_strategy(all_data, window=10, long_only=True)
    long_short = zscore_strategy(all_data, window=10, long_only=False)

    long_only_sign = sign_strategy(all_data, long_only=True)
    long_short_sign = sign_strategy(all_data, long_only=False)

    plt.plot((all_data['Close']/all_data['Close'].shift(1)-1).cumsum())
    plt.plot(long_only['cumulative_returns'])
    plt.plot(long_short['cumulative_returns'])
    plt.plot(long_only_sign['cumulative_returns'])
    plt.plot(long_short_sign['cumulative_returns'])
    plt.legend([ticker, "Long only mom","Long Short mom",
                "Long only sign","Long Short sign"])
    plt.show()