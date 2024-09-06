# THIS IS JARVIS.
#Dependencies
#TO_DO
#- sync across different exchanges 
#- sentiment analysis for news articcles feeding into volatility
#- reccomended sell, buy times 
#- more advances logic for take profit
#- check if stocks r available on exchange broker while fethcing 
#- more advanced fetch algo 
#- check balance and set order quantity accordingly
#- dashboard and email updates 
#- include covariance ( somehow ) 
#- twitter api for news too
#- news to make other decisions 
#- more indicators and advanced pattern recognition 
#- backtest and papertrade
#- monitor hourly and just efore close 
#- also on market open get all open orders and monitor 
# TRADING API HERE /////////////////////////////////////////////
import time
import logging
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import ccxt
import yfinance as yf
import requests
from alpaca_trade_api.rest import REST, TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, TakeProfitRequest, StopLossRequest, TrailingStopOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups
from dotenv import load_dotenv
import os
from pytz import timezone
import alpaca_trade_api as tradeapi
import pandas as pd
import logging
import time
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_API_SECRET = os.getenv('ALPACA_API_SECRET')
ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets/v2')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')

# Initialize logging
from logging.handlers import RotatingFileHandler
handler = RotatingFileHandler('trading_algorithm.log', maxBytes=5*1024*1024, backupCount=5)
logging.basicConfig(handlers=[handler], level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Alpaca API
api = REST(ALPACA_API_KEY, ALPACA_API_SECRET, base_url=ALPACA_BASE_URL)
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=True)

# Initialize data structures to track performance
portfolio = {}
trade_history = []
net_profit_loss = 0

# Initialize sentiment analysis model  ----- NEWS TRAINED MODEL (UPDATE WITH FINANCIAL DATA )
def initialize_sentiment_model():
    try:
        print("Initializing sentiment analysis model...")
        logging.info("Initializing sentiment analysis model...")
        categories = ['rec.sport.baseball', 'rec.sport.hockey']
        data = fetch_20newsgroups(subset='train', categories=categories)
        pipeline = Pipeline([
            ('vectorizer', CountVectorizer()),
            ('classifier', MultinomialNB())
        ])
        pipeline.fit(data.data, data.target)
        print("Sentiment analysis model initialized successfully.")
        logging.info("Sentiment analysis model initialized successfully.")
        return pipeline
    except Exception as e:
        print(f"Error initializing sentiment model: {e}")
        logging.error(f"Error initializing sentiment model: {e}")
        return None

sentiment_model = initialize_sentiment_model()



##############################################
#                Data Fetching               #
##############################################

#Fethces from wikipedia - need a better strategy to find stocks 
def fetch_trending_stocks(limit=10):
    """
    Fetch trending stocks based on highest volume from Yahoo Finance.
    """
    try:
        print("Fetching trending stocks...")
        logging.info("Fetching trending stocks...")
        # Fetch S&P 500 tickers
        sp500_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
        
        # Fetch market data for these tickers
        data = yf.download(sp500_tickers, period='1d', group_by='ticker', threads=True, progress=False)
        
        # Calculate average volume and sort
        avg_volumes = {}
        for ticker in sp500_tickers:
            try:
                avg_volume = data[ticker]['Volume'].mean()
                avg_volumes[ticker] = avg_volume
            except Exception as e:
                continue
        
        # Sort tickers by volume
        sorted_tickers = sorted(avg_volumes.items(), key=lambda x: x[1], reverse=True)
        trending_stocks = [ticker for ticker, volume in sorted_tickers[:limit]]
        
        print(f"Trending stocks fetched: {trending_stocks}")
        logging.info(f"Trending stocks fetched: {trending_stocks}")
        return trending_stocks
    except Exception as e:
        print(f"Error fetching trending stocks: {e}")
        logging.error(f"Error fetching trending stocks: {e}")
        return []


#Fetches market data for the stock
def fetch_data(symbol, timeframe='1D', limit=100):
    """
    Fetch historical market data for a given symbol and timeframe.
    """
    try:
        print(f"Fetching data for {symbol} with timeframe {timeframe}...")
        logging.info(f"Fetching data for {symbol} with timeframe {timeframe}...")
        timeframe_map = {
            '1D': TimeFrame.Day,
            '1H': TimeFrame.Hour,
            '15M': TimeFrame(15, TimeFrameUnit.Minute)
        }
        bars = api.get_bars(symbol, timeframe_map[timeframe], limit=limit).df
        bars = bars.tz_convert('US/Eastern')

        # if bars.empty:
        #     logging.warning(f"No data fetched for {symbol}.")
        #     return None
    
        print(f"Data fetched for {symbol}: {len(bars)} bars.")
        logging.info(f"Data fetched for {symbol}: {len(bars)} bars.")
        return bars
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        logging.error(f"Error fetching data for {symbol}: {e}")
        return None


#NEWS FOR THE STOCK
def fetch_news(symbol, from_date=None, to_date=None, language='en'):
    """
    Fetch recent news articles for a given stock symbol from News API.
    """
    try:
        print(f"Fetching news for {symbol}...")
        logging.info(f"Fetching news for {symbol}...")
        if from_date is None:
            from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        if to_date is None:
            to_date = datetime.now().strftime('%Y-%m-%d')
        
        url = (
            f"https://newsapi.org/v2/everything?"
            f"q={symbol}&from={from_date}&to={to_date}&language={language}&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
        )
        response = requests.get(url)
        response.raise_for_status()
        articles = response.json().get('articles', [])
        print(f"Fetched {len(articles)} articles for {symbol}.")
        logging.info(f"Fetched {len(articles)} articles for {symbol}.")
        return articles
    except Exception as e:
        print(f"Error fetching news for {symbol}: {e}")
        logging.error(f"Error fetching news for {symbol}: {e}")
        return []

##############################################
#            Technical Indicators            #
##############################################

def calculate_rsi(prices, period=14):
    """
    Calculate Relative Strength Index (RSI) for a series of prices.
    """
    try:
        print("Calculating RSI...")
        logging.info("Calculating RSI...")
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        print("RSI calculated.")
        logging.info("RSI calculated.")
        return rsi
    except Exception as e:
        print(f"Error calculating RSI: {e}")
        logging.error(f"Error calculating RSI: {e}")
        return pd.Series([])

def calculate_macd(prices):
    """
    Calculate Moving Average Convergence Divergence (MACD) for a series of prices.
    """
    try:
        print("Calculating MACD...")
        logging.info("Calculating MACD...")
        ema12 = prices.ewm(span=12, adjust=False).mean()
        ema26 = prices.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        print("MACD calculated.")
        logging.info("MACD calculated.")
        return macd, signal, histogram
    except Exception as e:
        print(f"Error calculating MACD: {e}")
        logging.error(f"Error calculating MACD: {e}")
        return pd.Series(), pd.Series(), pd.Series()
    
def calculate_atr(high, low, close, period=14, method='sma', default_atr_value=1.0, min_periods=5):
    """
    Calculate the Average True Range (ATR) for volatility measurement.

    :param high: Series containing the high prices.
    :param low: Series containing the low prices.
    :param close: Series containing the close prices.
    :param period: Period over which to calculate ATR (default is 14).
    :param method: Method to calculate ATR ('sma' for simple moving average, 'ema' for exponential moving average).
    :return: Series representing the ATR.
    """
    try:
        print("Calculating ATR...")
        logging.info("Calculating ATR...")
        if high.empty or low.empty or close.empty:
            logging.error("High, low, or close price data is empty, cannot calculate ATR.")
            return pd.Series([default_atr_value] * len(high))  # Return a default ATR series

        # Calculate True Range (TR)
        previous_close = close.shift(1)
        high_low = high - low
        high_prev_close = abs(high - previous_close)
        low_prev_close = abs(low - previous_close)
        tr = np.maximum(high_low, np.maximum(high_prev_close, low_prev_close))

        # Calculate ATR (SMA or EMA)
        if method == 'sma':
            atr = tr.rolling(window=period, min_periods=min_periods).mean()
        elif method == 'ema':
            atr = tr.ewm(span=period, adjust=False, min_periods=min_periods).mean()
        else:
            logging.error(f"Unknown method '{method}' specified for ATR calculation.")
            return pd.Series([default_atr_value] * len(high))  # Return default ATR if method is unknown

        # Fill NaN values with forward or backward filling
        atr.fillna(method='bfill', inplace=True)  # Use previous values if available
        atr.fillna(default_atr_value, inplace=True)  # Replace any remaining NaN with default

        # Optionally enforce a minimum ATR value (if necessary)
        atr = atr.apply(lambda x: max(x, default_atr_value))

        logging.info(f"ATR calculation completed with a default fallback. ATR: {atr}")
        return atr
    except Exception as e:
        print(f"Error calculating ATR: {e}")
        logging.error(f"Error calculating ATR: {e}")
        return pd.Series([])



def calculate_bollinger_bands(prices, window=20):
    """
    Calculate Bollinger Bands for a series of prices.
    """
    try:
        print("Calculating Bollinger Bands...")
        logging.info("Calculating Bollinger Bands...")
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        print("Bollinger Bands calculated.")
        logging.info("Bollinger Bands calculated.")
        return upper_band, sma, lower_band
    except Exception as e:
        print(f"Error calculating Bollinger Bands: {e}")
        logging.error(f"Error calculating Bollinger Bands: {e}")
        return pd.Series(), pd.Series(), pd.Series()

def calculate_indicators(data):
    """
    Calculate all necessary technical indicators and add them to the data DataFrame.
    """
    try:
        if data is None or 'close' not in data.columns:
            logging.warning("Data is empty or 'close' column is missing, skipping indicator calculation.")
            return data


        print("Calculating technical indicators...")
        logging.info("Calculating technical indicators...")
        df = data
        data = data.copy()
        data['RSI'] = calculate_rsi(data['close'])
        data['MACD'], data['MACD_Signal'], data['MACD_Hist'] = calculate_macd(data['close'])
        data['MA_50'] = data['close'].rolling(window=50).mean()
        data['MA_200'] = data['close'].rolling(window=200).mean()
        data['ATR'] = calculate_atr(data['high'], data['low'], data['close'], period=14, default_atr_value=1.5)
        data['Upper_Band'], data['Middle_Band'], data['Lower_Band'] = calculate_bollinger_bands(data['close'])
        print("Technical indicators calculated.")
        logging.info("Technical indicators calculated.")
        return data
    except Exception as e:
        print(f"Error calculating indicators: {e}")
        logging.error(f"Error calculating indicators: {e}")
        return data

##############################################
#          Advanced Pattern Detection        #
##############################################

def detect_divergence(data):
    """
    Detect bullish or bearish divergences between price and RSI.
    """
    try:
        print("Detecting divergences...")
        logging.info("Detecting divergences...")
        divergence_signals = []
        # Ensure sufficient data
        if len(data) < 30:
            return divergence_signals
        
        # Recent price and RSI changes
        price_diff = data['close'].iloc[-3:] - data['close'].iloc[-4:-1]
        rsi_diff = data['RSI'].iloc[-3:] - data['RSI'].iloc[-4:-1]
        
        # Bullish divergence: price makes lower lows, RSI makes higher lows
        if price_diff.iloc[-1] < 0 and rsi_diff.iloc[-1] > 0:
            divergence_signals.append('bullish divergence')
        
        # Bearish divergence: price makes higher highs, RSI makes lower highs
        if price_diff.iloc[-1] > 0 and rsi_diff.iloc[-1] < 0:
            divergence_signals.append('bearish divergence')
        
        print(f"Divergence signals: {divergence_signals}")
        logging.info(f"Divergence signals: {divergence_signals}")
        return divergence_signals
    except Exception as e:
        print(f"Error detecting divergence: {e}")
        logging.error(f"Error detecting divergence: {e}")
        return []

def detect_smt(data, related_symbol='SPY'):
    """
    Detect Smart Money Techniques (SMT) using price correlation with a related symbol.
    """
    try:
        print("Detecting Smart Money Techniques (SMT)...")
        logging.info("Detecting Smart Money Techniques (SMT)...")
        related_data = fetch_data(related_symbol, timeframe='1D', limit=100)
        if related_data is None:
            return []
        
        combined_data = pd.merge(data, related_data, left_index=True, right_index=True, suffixes=('', '_related'))
        correlation = combined_data['close'].corr(combined_data['close_related'])
        
        # Simple threshold-based detection
        if correlation < -0.5:
            print(f"Detected potential SMT: correlation={correlation:.2f}")
            logging.info(f"Detected potential SMT: correlation={correlation:.2f}")
            return ['smt']
        return []
    except Exception as e:
        print(f"Error detecting SMT: {e}")
        logging.error(f"Error detecting SMT: {e}")
        return []

def detect_fvg(data):
    """
    Detect Fair Value Gaps (FVG) in price data.
    """
    try:
        print("Detecting Fair Value Gaps (FVG)...")
        logging.info("Detecting Fair Value Gaps (FVG)...")
        fvg_signals = []
        if len(data) < 30:
            return fvg_signals
        
        gaps = data['close'].diff().dropna()
        for i in range(1, len(gaps)):
            if gaps.iloc[i] > gaps.iloc[i-1] * 1.05:
                fvg_signals.append('bullish gap')
            elif gaps.iloc[i] < gaps.iloc[i-1] * -1.05:
                fvg_signals.append('bearish gap')
        
        print(f"FVG signals: {fvg_signals}")
        logging.info(f"FVG signals: {fvg_signals}")
        return fvg_signals
    except Exception as e:
        print(f"Error detecting FVG: {e}")
        logging.error(f"Error detecting FVG: {e}")
        return []

##############################################
#               Trading Algorithm            #
##############################################

def generate_signals(data):
    """
    Generate trading signals based on technical indicators and patterns.
    """
    try:
        print("Generating trading signals...")
        logging.info("Generating trading signals...")
        signals = []
        data = calculate_indicators(data)

        # Check for the existence of the required columns
        if 'MA_50' not in data.columns or 'MA_200' not in data.columns or 'RSI' not in data.columns:
            logging.warning("Required columns for signal generation are missing, skipping.")
            return signals

        # Check moving average cross
        if data['MA_50'].iloc[-1] > data['MA_200'].iloc[-1]:
            signals.append('buy')
        elif data['MA_50'].iloc[-1] < data['MA_200'].iloc[-1]:
            signals.append('sell')
        
        # Check RSI levels
        if data['RSI'].iloc[-1] < 30:
            signals.append('buy')
        elif data['RSI'].iloc[-1] > 70:
            signals.append('sell')
        
        # Check MACD cross
        if data['MACD'].iloc[-1] > data['MACD_Signal'].iloc[-1]:
            signals.append('buy')
        elif data['MACD'].iloc[-1] < data['MACD_Signal'].iloc[-1]:
            signals.append('sell')
        
        # Check for divergences
        divergence_signals = detect_divergence(data)
        signals.extend(divergence_signals)
        
        # Check for SMT
        smt_signals = detect_smt(data)
        signals.extend(smt_signals)
        
        # Check for FVG
        fvg_signals = detect_fvg(data)
        signals.extend(fvg_signals)
        
        print(f"Generated signals: {signals}")
        logging.info(f"Generated signals: {signals}")
        return signals
    except Exception as e:
        print(f"Error generating signals: {e}")
        logging.error(f"Error generating signals: {e}")
        return []

# def execute_trade(signal, symbol, quantity=1):
#     """
#     Execute a trade based on the generated signal.
#     """
#     try:
#         print(f"Executing trade signal: {signal} for {symbol}")
#         logging.info(f"Executing trade signal: {signal} for {symbol}")
#         if signal == 'buy':
#             api.submit_order(
#                 symbol=symbol,
#                 qty=quantity,
#                 side='buy',
#                 type='market',
#                 time_in_force='gtc'
#             )
#             print(f"Executed buy order for {symbol}.")
#             logging.info(f"Executed buy order for {symbol}.")
#         elif signal == 'sell':
#             api.submit_order(
#                 symbol=symbol,
#                 qty=quantity,
#                 side='sell',
#                 type='market',
#                 time_in_force='gtc'
#             )
#             print(f"Executed sell order for {symbol}.")
#             logging.info(f"Executed sell order for {symbol}.")
#         else:
#             print(f"No valid trade signal for {symbol}.")
#             logging.info(f"No valid trade signal for {symbol}.")
#     except Exception as e:
#         print(f"Error executing trade for {symbol}: {e}")
#         logging.error(f"Error executing trade for {symbol}: {e}")

def update_portfolio(symbol, action, quantity):
    """
    Update the portfolio with the latest trade details.
    """
    global portfolio
    try:
        if symbol not in portfolio:
            portfolio[symbol] = {'quantity': 0, 'average_price': 0}
        
        if action == 'buy':
            current_price = yf.Ticker(symbol).info['last_price']
            total_cost = portfolio[symbol]['average_price'] * portfolio[symbol]['quantity']
            new_quantity = portfolio[symbol]['quantity'] + quantity
            new_average_price = (total_cost + current_price * quantity) / new_quantity
            portfolio[symbol]['quantity'] = new_quantity
            portfolio[symbol]['average_price'] = new_average_price
            print(f"Updated portfolio: Bought {quantity} of {symbol} at average price {new_average_price}.")
            logging.info(f"Updated portfolio: Bought {quantity} of {symbol} at average price {new_average_price}.")
        elif action == 'sell':
            portfolio[symbol]['quantity'] -= quantity
            if portfolio[symbol]['quantity'] == 0:
                del portfolio[symbol]
            print(f"Updated portfolio: Sold {quantity} of {symbol}.")
            logging.info(f"Updated portfolio: Sold {quantity} of {symbol}.")
    except Exception as e:
        print(f"Error updating portfolio for {symbol}: {e}")
        logging.error(f"Error updating portfolio for {symbol}: {e}")

def trade_loop():
    """
    Main trading loop to continuously monitor and trade based on signals.
    """
    print("JARVIS IS ALGO TRADING")
    logging.info("JARVIS IS ALGO TRADING")
    global trade_history
    while True:
        try:
            # Fetch trending stocks
            trending_stocks = fetch_trending_stocks()
            
            for symbol in trending_stocks:
                # Fetch and analyze data
                data = fetch_data(symbol, timeframe='1D', limit=100)

                if data is None:
                    continue
                
                # Generate signals
                signals = generate_signals(data)
                
                for signal in signals:
                    # Execute trades based on signals
                    execute_trade(signal, symbol)
                    update_portfolio(symbol, signal, 1)  # Adjust quantity as needed
                    trade_history.append({'timestamp': datetime.now(), 'symbol': symbol, 'signal': signal})
                    print(f"Trade executed: {signal} for {symbol}")
                    logging.info(f"Trade executed: {signal} for {symbol}")
                    
                # Fetch and analyze news
                news_articles = fetch_news(symbol)
                if sentiment_model:
                    news_sentiments = [sentiment_model.predict([article['title'] + ' ' + article['description']])[0] for article in news_articles]
                    sentiment_score = np.mean(news_sentiments)
                    print(f"Sentiment score for {symbol}: {sentiment_score}")
                    logging.info(f"Sentiment score for {symbol}: {sentiment_score}")
                
            time.sleep(60)  # Run every minute
        except Exception as e:
            print(f"Error in trading loop: {e}")
            logging.error(f"Error in trading loop: {e}")
            time.sleep(60)


def fetch_historical_data_yfin(symbol, timeframe='1D', limit=100):
    try:
        print(f"Fetching {timeframe} historical data for {symbol}...")
        logging.info(f"Fetching {timeframe} historical data for {symbol}...")

        # Mapping the timeframe to the appropriate yfinance interval
        interval_map = {
            '1D': '1d',
            '15Min': '15m',
            '1H': '1h',
            '1W': '1wk',
            '1M': '1mo'
        }

        if timeframe not in interval_map:
            raise ValueError(f"Timeframe {timeframe} is not supported.")

        interval = interval_map[timeframe]

        # Fetch historical data using yfinance
        df = yf.download(symbol, period=f"{limit}d", interval=interval)

        if isinstance(df, pd.DataFrame) and not df.empty:
            print(f"Data fetched successfully for {symbol}.")
            logging.info(f"Data fetched successfully for {symbol}.")
            return df
        else:
            print(f"No data found for {symbol}.")
            logging.warning(f"No data found for {symbol}.")
            return pd.DataFrame()

    except Exception as e:
        print(f"Error fetching historical data for {symbol}: {e}")
        logging.error(f"Error fetching historical data for {symbol}: {e}")
        return pd.DataFrame()
    
def fetch_historical_data_alpaca(symbol, timeframe='1D', limit=100):
    try:
        print(f"Fetching {timeframe} historical data for {symbol}...")
        logging.info(f"Fetching {timeframe} historical data for {symbol}...")
        
        # # Fetch historical data from Alpaca
        # barset = api.get_bars(symbol, timeframe, limit=limit)
        # bars = barset[symbol]

        # # Convert to DataFrame
        # data = {
        #     'time': [bar.t for bar in bars],
        #     'open': [bar.o for bar in bars],
        #     'high': [bar.h for bar in bars],
        #     'low': [bar.l for bar in bars],
        #     'close': [bar.c for bar in bars],
        #     'volume': [bar.v for bar in bars],
        # }
        # df = pd.DataFrame(data)
        # df.set_index('time', inplace=True)

        # return df
        bars = api.get_bars(symbol, timeframe, limit=limit).df
        # Verify that data is returned as a DataFrame
        # if isinstance(bars, pd.DataFrame):
            # return bars
        
        print(f"Data fetched successfully for {symbol}.")
        logging.info(f"Data fetched successfully for {symbol}.")

        return bars
        # else:
            # print(f"Unexpected data format for {symbol}: {type(bars)}")
            # return None

    except Exception as e:
        print(f"Error fetching historical data for {symbol}: {e}")
        logging.error(f"Error fetching historical data for {symbol}: {e}")
        # return pd.DataFrame()
        return None

def analyze_higher_timeframes(symbol):
    try:
        print("Analyzing higher timeframes...")
        logging.info("Analyzing higher timeframes...")
        
        df = fetch_historical_data_alpaca(symbol, timeframe='1D', limit=100)

        # print("df", df)
        
        if df.empty:
            return None, pd.DataFrame()

        df = calculate_indicators(df)

        df['trend'] = df['MA_50'] > df['MA_200']
        df['support'] = df['low'].rolling(window=50).min()
        df['resistance'] = df['high'].rolling(window=50).max()

        if df['trend'].iloc[-1]:
            bias = 'Bullish' if df['close'].iloc[-1] < df['resistance'].iloc[-1] else 'Bearish'
        else:
            bias = 'Bearish' if df['close'].iloc[-1] > df['support'].iloc[-1] else 'Bullish'

        print(f"Market bias determined: {bias}")
        logging.info(f"Market bias determined: {bias}")
        return bias, df
    except Exception as e:
        print(f"Error analyzing higher timeframes: {e}")
        logging.error(f"Error analyzing higher timeframes: {e}")
        return None, pd.DataFrame()

def analyze_lower_timeframes(symbol, bias):
    try:
        print("Analyzing lower timeframes...")
        logging.info("Analyzing lower timeframes...")
        
        df = fetch_historical_data_alpaca(symbol, timeframe='15Min', limit=100)
        
        if df.empty:
            return []

        df = calculate_indicators(df)        
        data = df.copy()
        rsi = calculate_rsi(data['close'])
        signals = generate_signals(df)
        
        filtered_signals = []
        for signal in signals:
            if (bias == 'Bullish' and signal == 'buy') or (bias == 'Bearish' and signal == 'sell'):
                filtered_signals.append(signal)
        
        print(f"Filtered signals based on bias: {filtered_signals}")
        logging.info(f"Filtered signals based on bias: {filtered_signals}")
        return filtered_signals, rsi
    except Exception as e:
        print(f"Error analyzing lower timeframes: {e}")
        logging.error(f"Error analyzing lower timeframes: {e}")
        return []


def pre_market_analysis(symbol):
    # Analyze higher timeframes
    bias, higher_timeframe_data = analyze_higher_timeframes(symbol)

        # Check if ATR was calculated
    if 'ATR' not in higher_timeframe_data.columns:
        print("ATR was not calculated.")
        logging.error("ATR was not calculated.")
        return None, None

    if not bias:
        print(f"Could not determine market bias for {symbol}.")
        return None, None

    # Analyze lower timeframes and set trade price points
    lower_timeframe_signals, rsi = analyze_lower_timeframes(symbol, bias)

    # Ensure ATR is usable and non-NaN
    atr_value = higher_timeframe_data['ATR'].iloc[-1]
    if pd.isna(atr_value):
        print("ATR value is NaN, unable to calculate risk.")
        logging.error("ATR value is NaN, unable to calculate risk.")
        return None, None

    print(f"higher_timeframe_data['ATR']: {atr_value}")
    
    # Calculate volatility and risk management values
    volatility = atr_value

    volatility = higher_timeframe_data['ATR'].iloc[-1]  # Example: Using ATR for volatility
    print(f"volatility: {volatility}, rt: 0.02")

    risk_percentage = calculate_risk_percentage(volatility, 0.02)  # Example risk tolerance of 2%
    reward_to_risk_ratio = calculate_reward_to_risk_ratio(bias)  # Example reward-to-risk ratio


    trade_price_points = []
    for signal in lower_timeframe_signals:
        if signal == 'buy':
            entry_price = higher_timeframe_data['close'].iloc[-1]
            print("entry :  ", entry_price, "rp : ", risk_percentage)
            stop_loss = calculate_stop_loss(entry_price, risk_percentage)
            take_profit = calculate_take_profit(entry_price, reward_to_risk_ratio, stop_loss)
            print("sl :  ", stop_loss, "tp : ", take_profit)
            trade_price_points.append({
                'action': 'buy',
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'use_trailing_stop': should_use_trailing_stop(higher_timeframe_data, bias)
            })
        elif signal == 'sell':
            entry_price = higher_timeframe_data['close'].iloc[-1]
            print("entry :  ", entry_price, "rp : ", risk_percentage)
            stop_loss = calculate_stop_loss(entry_price, risk_percentage)
            take_profit = calculate_take_profit(entry_price, reward_to_risk_ratio, stop_loss)
            print("sl :  ", stop_loss, "tp : ", take_profit)
            trade_price_points.append({
                'action': 'sell',
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'use_trailing_stop': should_use_trailing_stop(higher_timeframe_data, bias)
            })

    return bias, trade_price_points, rsi

def calculate_stop_loss(entry_price, risk_percentage):
    """Calculate stop loss based on risk percentage."""
    stop_loss = entry_price * (1 - risk_percentage)
    return round(stop_loss, 2)

def calculate_take_profit(entry_price, reward_to_risk_ratio, stop_loss):
    """Calculate take profit based on reward-to-risk ratio."""
    take_profit = entry_price + (entry_price - stop_loss) * reward_to_risk_ratio
    return round(take_profit, 2)

def should_use_trailing_stop(df, bias, atr_threshold=1.5):
    """
    Decide whether to use a trailing stop based on volatility and market bias.
    :param df: DataFrame containing historical data with calculated ATR.
    :param bias: Market bias ('Bullish' or 'Bearish').
    :param atr_threshold: Threshold for ATR to determine high volatility.
    :return: Boolean indicating whether to use a trailing stop.
    """
    atr = df['ATR'].iloc[-1]
    if bias == 'Bullish' and atr > atr_threshold:
        return True
    elif bias == 'Bearish' and atr > atr_threshold:
        return True
    return False


def calculate_reward_to_risk_ratio(market_condition):
    """
    Calculate reward-to-risk ratio based on market conditions.
    :param market_condition: Indicator of current market condition (e.g., trend strength).
    :return: Reward-to-risk ratio.
    """
    if market_condition == "bullish":
        return 3  # Favor higher reward in bullish markets
    elif market_condition == "bearish":
        return 1.5  # Favor lower reward in bearish markets
    else:
        return 2  # Default value


def calculate_risk_percentage(volatility, risk_tolerance=0.02):
    """
    Calculate risk percentage based on market volatility and risk tolerance.
    :param volatility: Standard deviation or ATR (Average True Range) representing market volatility.
    :param risk_tolerance: Base risk tolerance as a percentage of the entry price (default is 2%).
    :return: Adjusted risk percentage.
    """
    # Adjust risk based on volatility
    adjusted_risk = risk_tolerance * volatility
    # Ensure risk percentage does not exceed a certain limit, e.g., 5%
    return min(adjusted_risk, 0.05)


def execute_trades(symbol, trade_points, rsi, bias):
    print("executing trades")
    for trade in trade_points:
     action = trade['action']
     entry_price = trade['entry_price']
     stop_loss = trade['stop_loss']
     take_profit = trade['take_profit']
     use_trailing_stop = trade['use_trailing_stop']
        
     logging.info(f"Prepared {action.upper()} order for {symbol}: Entry={entry_price}, Stop Loss={stop_loss}, "
                     f"Take Profit={take_profit}, Use Trailing Stop={use_trailing_stop}")
     print(f"Prepared {action.upper()} order for {symbol}: Entry={entry_price}, Stop Loss={stop_loss}, "
                     f"Take Profit={take_profit}, Use Trailing Stop={use_trailing_stop}")
    
     try:
        # Place conditional orders
        if action == 'buy':
            if use_trailing_stop:
                trailing_stop_order_data = TrailingStopOrderRequest(
                    symbol=symbol,
                    qty=1,  # Adjust quantity based on your position sizing logic
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.GTC,
                    extended_hours=True,
                    trail_price=1.00  # Example trailing stop value
                )
                trailing_stop_order = trading_client.submit_order(
                    order_data=trailing_stop_order_data
                )
            else:
                bracket_order_data = MarketOrderRequest(
                    symbol=symbol,
                    qty=1,  # Adjust quantity based on your position sizing logic
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.GTC,
                    order_class=OrderClass.BRACKET,
                    extended_hours=True,
                    take_profit=TakeProfitRequest(limit_price=take_profit),
                    stop_loss=StopLossRequest(stop_price=stop_loss)
                )
                bracket_order = trading_client.submit_order(
                    order_data=bracket_order_data
                )

        elif action == 'sell':
            if use_trailing_stop:
                trailing_stop_order_data = TrailingStopOrderRequest(
                    symbol=symbol,
                    qty=1,  # Adjust quantity based on your position sizing logic
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC,
                    extended_hours=True,
                    trail_price=1.00  # Example trailing stop value
                )
                trailing_stop_order = trading_client.submit_order(
                    order_data=trailing_stop_order_data
                )
            else:
                bracket_order_data = MarketOrderRequest(
                    symbol=symbol,
                    qty=1,  # Adjust quantity based on your position sizing logic
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC,
                    order_class=OrderClass.BRACKET,
                    extended_hours=True,
                    take_profit=TakeProfitRequest(limit_price=take_profit),
                    stop_loss=StopLossRequest(stop_price=stop_loss)
                )
                bracket_order = trading_client.submit_order(
                    order_data=bracket_order_data
                )

            # Log the details of the order
            print(f"Placed {action.upper()} order for {symbol}: Entry={entry_price}, Stop Loss={stop_loss}, Take Profit={take_profit}")
     except requests.exceptions.InvalidJSONError as e:
            logging.error(f"Failed to place order due to invalid JSON: {e}")
     except Exception as e:
            logging.error(f"An error occurred: {e}")

def main_trading_routine():
    nyc = timezone('America/New_York')
    market_open_time = nyc.localize(datetime.combine(datetime.today(), datetime.strptime("09:30:00", "%H:%M:%S").time()))
    pre_market_time = market_open_time - timedelta(hours=1)

    # Wait until one hour before market opens
    # while nyc.localize(datetime.now()) < pre_market_time:
    #     time.sleep(60)  # Sleep for 1 minute

    # Fetch trending stocks
    trending_stocks = fetch_trending_stocks()

    # Pre-market analysis
    trade_setups = {}
    for symbol in trending_stocks:
        bias, trade_points, rsi = pre_market_analysis(symbol)
        if trade_points:
            trade_setups[symbol] = trade_points

    # Wait for the market to open
    # while nyc.localize(datetime.now()) < market_open_time:
    #     time.sleep(60)  # Sleep for 1 minute

    # Execute trades based on pre-market setups
    for symbol, trade_points in trade_setups.items():
        execute_trades(symbol, trade_points, rsi, bias)

    # Continuous in-market analysis and execution
    # while True:
    # print(trade_setups)
    # for symbol in trending_stocks:
            # if symbol in trade_setups:
                # lower_timeframe_signals, rsi = analyze_lower_timeframes(symbol, trade_setups[symbol][0]['action'])
                
            # Execute trades or adjust positions based on new analysis
            # for signal in lower_timeframe_signals:
                # if signal == 'buy' or signal == 'sell':

                # for signal in lower_timeframe_signals:
                    # if signal in ['buy', 'sell']:
                        # execute_trades(symbol, trade_setups[symbol], rsi, bias)


        # Wait for the next interval (e.g., 15 minutes)
    # time.sleep(900)  # 15 minutes

    # while True:
    # for symbol in trending_stocks:
    #     if symbol in trade_setups:
    #         lower_timeframe_signals, rsi = analyze_lower_timeframes(symbol, trade_setups[symbol][0]['action'])
            
    #         # Execute trades or adjust positions based on new analysis
    #         execute_trades(symbol, trade_setups[symbol], rsi, trade_setups[symbol][0]['action'])

    # # Wait for the next interval (e.g., 15 minutes)
    # time.sleep(900)  # 15 minutes

# Start the trading routine
if __name__ == "__main__":
    main_trading_routine()


# if __name__ == "__main__":
#     trade_loop()


# TWITTER API HERE  ////////////////////////////////////////////

# LINKEDINN HERE /////////////////////////////////////////////////

# LEAD GEN HERE ////////////////////////////////////////////////////


