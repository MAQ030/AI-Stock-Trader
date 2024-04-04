from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from datetime import datetime, timedelta  # Import statement corrected

from alpaca_trade_api import REST
from finbert_utils import estimate_sentiment

API_KEY = "PK1P28OTG8GV0LVJPCNL"
API_SECRET = "nRnALBIz3zF6QwIjyUv3WmN4yAngQ169FBxpev91"
BASE_URL = "https://paper-api.alpaca.markets/v2"

ALPACA_CREDS = {
    "API_KEY": API_KEY,
    "API_SECRET": API_SECRET,
    "PAPER": True  # Indicates this is a paper trading account, update if using real money
}

class MLTrader(Strategy):
    def initialize(self, symbol="SPY", cash_at_risk=0.5):
        """
        Initializes the MLTrader strategy.

        Parameters:
        - symbol (str): Symbol to trade, default is "SPY".
        - cash_at_risk (float): Percentage of cash to risk per trade, default is 0.5.
        """
        self.symbol = symbol
        self.sleeptime = "24H"  # Consider revising this for more dynamic behavior
        self.last_trade = None
        self.cash_at_risk = cash_at_risk
        self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)

    def position_sizing(self):
        """
        Calculates position size based on risk management rules.

        Returns:
        - cash (float): Available cash.
        - last_price (float): Last price of the symbol.
        - quantity (int): Calculated quantity to trade.
        """
        cash = self.get_cash()
        last_price = self.get_last_price(self.symbol)
        quantity = round(cash * self.cash_at_risk / last_price, 0)
        return cash, last_price, quantity

    def get_dates(self):
        """
        Calculates the date range for sentiment analysis.

        Returns:
        - today (str): Today's date in 'YYYY-MM-DD' format.
        - three_days_prior (str): Three days prior to today's date in 'YYYY-MM-DD' format.
        """
        today = self.get_datetime()
        three_days_prior = today - timedelta(days=3)  # Updated timedelta usage
        return today.strftime('%Y-%m-%d'), three_days_prior.strftime('%Y-%m-%d')

    def get_sentiment(self):
        """
        Fetches news and estimates sentiment for the symbol.

        Returns:
        - probability (float): Probability score of sentiment.
        - sentiment (str): Sentiment label ('positive', 'negative', 'neutral').
        """
        today, three_days_prior = self.get_dates()
        news = self.api.get_news(symbol=self.symbol, start=three_days_prior, end=today)
        news = [ev.__dict__["_raw"]["headline"] for ev in news]
        probability, sentiment = estimate_sentiment(news)
        return probability, sentiment

    def on_trading_iteration(self):
        """
        Executes trading logic based on sentiment and risk management.
        """
        cash, last_price, quantity = self.position_sizing()
        probability, sentiment = self.get_sentiment()
        
        if cash > last_price: 
            if sentiment == "positive" and probability > .999: 
                if self.last_trade == "sell": 
                    self.sell_all() 
                order = self.create_order(
                    self.symbol, 
                    quantity, 
                    "buy", 
                    type="bracket", 
                    take_profit_price=last_price*1.20, 
                    stop_loss_price=last_price*.95
                )
                self.submit_order(order) 
                self.last_trade = "buy"
            elif sentiment == "negative" and probability > .999: 
                if self.last_trade == "buy": 
                    self.sell_all() 
                order = self.create_order(
                    self.symbol, 
                    quantity, 
                    "sell", 
                    type="bracket", 
                    take_profit_price=last_price*.8, 
                    stop_loss_price=last_price*1.05
                )
                self.submit_order(order) 
                self.last_trade = "sell"

# Define the backtesting parameters
start_date = datetime(2022, 1, 1)  # Start date for backtesting, adjust as needed
end_date = datetime(2023, 12, 31)  # End date for backtesting, adjust as needed

# Initialize Alpaca broker with provided credentials
broker = Alpaca(ALPACA_CREDS)

# Initialize MLTrader strategy with parameters
strategy = MLTrader(name='mlstrat', broker=broker,
                    parameters={"symbol": "SPY", 
                                "cash_at_risk": 0.5})  # Adjust symbol and risk as needed

# Perform backtesting with YahooDataBacktesting
strategy.backtest(
    YahooDataBacktesting,
    start_date,
    end_date,
    parameters={"symbol": "SPY", "cash_at_risk": 0.5}  # Adjust symbol and risk as needed
)
