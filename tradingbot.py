from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from datetime import datetime, timedelta

from alpaca_trade_api import REST
from sentimentAnalyzer import estimate_sentiment

# Constants for API credentials and URL (you need to replace placeholders with actual values).
API_KEY = "API_KEY"
API_SECRET = "API_SECRET"
BASE_URL = "BASE_URL"

# Credentials for Alpaca API, specifying that it's a paper trading account.
ALPACA_CREDS = {
    "API_KEY": API_KEY,
    "API_SECRET": API_SECRET,
    "PAPER": True
}

class MLTrader(Strategy):
    """
    A trading strategy class that utilizes machine learning to decide when to buy or sell stocks based on sentiment analysis.
    """
    def initialize(self, symbol="SPY", cash_at_risk=0.5):
        """
        Initializes the MLTrader strategy with the trading symbol and the amount of cash to risk.
        """
        self.symbol = symbol
        self.sleeptime = "24H"  # Time between each trading decision. Modify for more frequent trading decisions.
        self.last_trade = None
        self.cash_at_risk = cash_at_risk
        self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)

    def position_sizing(self):
        """
        Calculates the quantity of stocks to buy or sell based on the available cash and risk management rules.
        """
        cash = self.get_cash()
        last_price = self.get_last_price(self.symbol)
        quantity = int(cash * self.cash_at_risk / last_price)  # Use integer quantity for orders
        return cash, last_price, quantity

    def get_dates(self):
        """
        Determines the date range for conducting sentiment analysis.
        """
        today = self.get_datetime()
        three_days_prior = today - timedelta(days=3)
        return today.strftime('%Y-%m-%d'), three_days_prior.strftime('%Y-%m-%d')

    def get_sentiment(self):
        """
        Analyzes news headlines to determine market sentiment toward the trading symbol.
        """
        today, three_days_prior = self.get_dates()
        news = self.api.get_news(symbol=self.symbol, start=three_days_prior, end=today)
        headlines = [event.__dict__["_raw"]["headline"] for event in news]
        probability, sentiment = estimate_sentiment(headlines)
        return probability, sentiment

    def on_trading_iteration(self):
        """
        Executed for each trading iteration, makes buy or sell decisions based on sentiment analysis.
        """
        cash, last_price, quantity = self.position_sizing()
        probability, sentiment = self.get_sentiment()
        
        if cash > last_price: 
            if sentiment == "positive" and probability > 0.999: 
                if self.last_trade == "sell": 
                    self.sell_all()
                order = self.create_order(
                    self.symbol, quantity, "buy", type="bracket",
                    take_profit_price=last_price*1.20,
                    stop_loss_price=last_price*0.95
                )
                self.submit_order(order)
                self.last_trade = "buy"
            elif sentiment == "negative" and probability > 0.999: 
                if self.last_trade == "buy": 
                    self.sell_all()
                order = self.create_order(
                    self.symbol, quantity, "sell", type="bracket",
                    take_profit_price=last_price*0.80,
                    stop_loss_price=last_price*1.05
                )
                self.submit_order(order)
                self.last_trade = "sell"

# Define backtesting parameters with start and end dates
start_date = datetime(2020, 1, 1)
end_date = datetime(2024, 4, 3)

# Set up the broker and strategy
broker = Alpaca(ALPACA_CREDS)
strategy = MLTrader(name='MLTrader', broker=broker, parameters={"symbol": "SPY", "cash_at_risk": 0.5})

# Execute backtesting using Yahoo data
strategy.backtest(
    YahooDataBacktesting,
    start_date,
    end_date,
    parameters={"symbol": "SPY", "cash_at_risk": 0.5}
)
