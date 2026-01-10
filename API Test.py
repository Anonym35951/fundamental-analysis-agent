import yfinance as yf
stock = yf.Ticker("AAPL")
financials = stock.financials
print(financials)