from pybit.unified_trading import HTTP

class BybitEx:
    def __init__(self, api_key: str, api_secret: str, testnet: bool):
        self.http = HTTP(api_key=api_key, api_secret=api_secret, testnet=testnet)

    def get_klines(self, symbol: str, interval: str, start_ms: int, end_ms: int, category="linear"):
        return self.http.get_kline(category=category, symbol=symbol, interval=interval, start=start_ms, end=end_ms)

    # בהמשך: place_order, cancel_all, get_positions, וכד'
