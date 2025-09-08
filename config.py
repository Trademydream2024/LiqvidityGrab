import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class _CFG:
    TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_TOKEN", "")
    BYBIT_API_KEY: str = os.getenv("BYBIT_API_KEY", "")
    BYBIT_API_SECRET: str = os.getenv("BYBIT_API_SECRET", "")
    BYBIT_TESTNET: bool = os.getenv("BYBIT_TESTNET", "true").lower() == "true"
    TZ: str = os.getenv("TIMEZONE", "Asia/Jerusalem")
    DATA_DIR: str = os.path.join(os.getcwd(), "data")

CFG = _CFG()
