from aiogram.utils.keyboard import InlineKeyboardBuilder

def kb_back_home():
    kb = InlineKeyboardBuilder()
    kb.button(text="🏠 בית", callback_data="nav:home")
    return kb.as_markup()

def kb_yes_no(cb_yes: str, cb_no: str):
    kb = InlineKeyboardBuilder()
    kb.button(text="✅ כן", callback_data=cb_yes)
    kb.button(text="❌ לא", callback_data=cb_no)
    kb.adjust(2)
    return kb.as_markup()
