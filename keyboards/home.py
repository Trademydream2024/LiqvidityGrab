from aiogram.utils.keyboard import InlineKeyboardBuilder

def kb_home():
    kb = InlineKeyboardBuilder()
    kb.button(text="📥 הורדת דאטה", callback_data="nav:data")
    kb.button(text="📊 בק־טסט", callback_data="nav:bt")
    kb.button(text="🔧 אופטימיזציה", callback_data="nav:opt")
    kb.button(text="🔴 פאנל לייב", callback_data="nav:live")
    kb.adjust(2,2)
    return kb.as_markup()
