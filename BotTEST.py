import sys
import os

print("=" * 50)
print("🔍 בודק את מערכת הבוט...")
print("=" * 50)

# 1. בדיקת חבילות
print("\n📦 בודק חבילות Python...")
missing_packages = []

try:
    import aiogram
    print("✅ aiogram מותקן")
except ImportError:
    print("❌ aiogram חסר")
    missing_packages.append("aiogram")

try:
    import pandas
    print("✅ pandas מותקן")
except ImportError:
    print("❌ pandas חסר")
    missing_packages.append("pandas")

try:
    import numpy
    print("✅ numpy מותקן")
except ImportError:
    print("❌ numpy חסר")
    missing_packages.append("numpy")

try:
    import pytz
    print("✅ pytz מותקן")
except ImportError:
    print("❌ pytz חסר")
    missing_packages.append("pytz")

try:
    import pybit
    print("✅ pybit מותקן")
except ImportError:
    print("⚠️ pybit לא מותקן (אופציונלי - רק אם רוצה להוריד דאטה חדש)")

if missing_packages:
    print(f"\n❗ להתקנת החבילות החסרות, הרץ:")
    print(f"pip install {' '.join(missing_packages)}")

# 2. בדיקת מבנה תיקיות
print("\n📁 בודק מבנה תיקיות...")
required_dirs = ['handlers', 'services', 'keyboards', 'utils', 'data']
for dir_name in required_dirs:
    if os.path.exists(dir_name):
        print(f"✅ {dir_name}/ קיים")
    else:
        print(f"❌ {dir_name}/ חסר - יוצר...")
        os.makedirs(dir_name)

# 3. בדיקת קבצים חשובים
print("\n📄 בודק קבצים חשובים...")
required_files = [
    'run.py',
    'config.py',
    'handlers/home.py',
    'handlers/data.py',
    'handlers/backtest.py',
    'handlers/optimize.py',
    'services/backtester.py',
    'services/strategy.py'
]

missing_files = []
for file in required_files:
    if os.path.exists(file):
        print(f"✅ {file}")
    else:
        print(f"❌ {file} חסר")
        missing_files.append(file)

# 4. בדיקת config.py
print("\n⚙️ בודק הגדרות...")
try:
    from config import CFG
    if CFG.TELEGRAM_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("⚠️ צריך להגדיר TELEGRAM_TOKEN ב-config.py!")
    else:
        print(f"✅ TELEGRAM_TOKEN מוגדר ({CFG.TELEGRAM_TOKEN[:10]}...)")
    
    print(f"📁 DATA_DIR: {CFG.DATA_DIR}")
    print(f"🌍 TZ: {CFG.TZ}")
except Exception as e:
    print(f"❌ בעיה ב-config.py: {e}")

# 5. בדיקת קבצי דאטה
print("\n💾 בודק קבצי דאטה...")
if os.path.exists('data'):
    parquet_files = [f for f in os.listdir('data') if f.endswith('.parquet')]
    if parquet_files:
        print(f"✅ נמצאו {len(parquet_files)} קבצי parquet:")
        for f in parquet_files[:5]:  # מציג עד 5
            size_mb = os.path.getsize(f'data/{f}') / (1024*1024)
            print(f"   • {f} ({size_mb:.2f} MB)")
            
            # נסיון לקרוא את הקובץ
            try:
                import pandas as pd
                df = pd.read_parquet(f'data/{f}')
                print(f"     ✓ {len(df):,} שורות, עמודות: {list(df.columns)[:5]}")
            except Exception as e:
                print(f"     ✗ לא ניתן לקרוא: {e}")
    else:
        print("⚠️ אין קבצי parquet בתיקיית data/")
else:
    print("❌ תיקיית data/ לא קיימת")

# סיכום
print("\n" + "=" * 50)
if not missing_packages and not missing_files:
    print("✅ המערכת מוכנה להרצה!")
    print("\nכדי להפעיל את הבוט:")
    print("python run.py")
else:
    print("⚠️ יש בעיות שצריך לתקן לפני הרצה")
    if missing_packages:
        print(f"\n1. התקן חבילות חסרות:")
        print(f"   pip install {' '.join(missing_packages)}")
    if missing_files:
        print(f"\n2. קבצים חסרים: {', '.join(missing_files)}")

print("=" * 50)