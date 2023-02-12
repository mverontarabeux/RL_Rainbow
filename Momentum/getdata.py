import pandas_datareader.data as web
import pandas as pd
import yfinance as yfin

class FinancialDataProvider:
    def __init__(self):
        return None

    def get_ticker(self, tickers, start_date, end_date, clean_it=True):
        if isinstance(tickers, str):
            tickers = [tickers]
        assert isinstance(tickers, list)
        yfin.pdr_override()
        
        data = web.get_data_yahoo(tickers, start_date, end_date)

        if clean_it:
            data = self.clean_data(data)

        return data

    def clean_data(self, data, columns=["Close"]):
        if len(columns)==1:
            # Get the relevant columns only
            relevant_data = data[columns[0]]
            relevant_data = pd.DataFrame(relevant_data)
        elif len(columns)>1:
            relevant_data = data[columns]
        else:
            return False

        # Get all the weekdays
        first_date = relevant_data.index[0]
        last_date = relevant_data.index[-1]
        all_weekdays = pd.date_range(start=first_date, end=last_date, freq='B')
        
        # Put the all weekdays as index 
        relevant_data = relevant_data.reindex(all_weekdays)
        relevant_data = relevant_data.fillna(method='ffill')
        return relevant_data
        

if __name__=='__main__':
    import matplotlib.pyplot as plt
    
    # The tickers list
    tickers = ['SPY']

    # Set the dates
    start_date = '2000-01-02'
    end_date = '2023-02-08'

    # load the data
    data_provider = FinancialDataProvider()
    all_data = data_provider.get_ticker(tickers, start_date, end_date)

    # Plot the results
    all_data.plot()
    plt.show()
