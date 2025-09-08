# utils/state.py
from typing import Dict

_CHAT_MODE: Dict[int, str] = {}  # chat_id -> mode

def set_mode(chat_id: int, mode: str) -> None:
    _CHAT_MODE[chat_id] = mode

def get_mode(chat_id: int) -> str:
    return _CHAT_MODE.get(chat_id, "home")
