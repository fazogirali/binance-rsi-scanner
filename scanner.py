"""
Binance RSI + Divergensiya skaner - 4-bosqich (faqat XARID uchun)
-----------------------------------------------------------------
Binance'dagi USDT juftliklarni H1, H4, D1 taymfreymlarida tekshiradi.
Faqat YOPILGAN shamlar hisobga olinadi.

  1) 🔻 RSI 30 dan PASTDA yopildi       (oldingi sham 30 dan yuqorida edi)
  2) 📍 RSI impuls bilan 30 ni kesdi    (sham ichida narx pastga sho'ng'ib RSI 30 dan
                                         tushdi, lekin sham tepada yopildi)
  3) 🔼 RSI 30 dan YUQORIGA qaytdi
  4) 🟢 Bullish divergensiya (shakllanmoqda / tasdiqlandi)

Ishga tushirish:
    python scanner.py test       -> Telegram ulanishini tekshirish
    python scanner.py auto all   -> GitHub uchun: har soat TO'LIQ HISOBOT (yangi yopilgan shamlar bo'yicha)
    python scanner.py auto       -> GitHub uchun: har soat, faqat YANGI hodisalar (qisqa)
    python scanner.py            -> qo'lda: barcha taymfreymlarning oxirgi shami
    python scanner.py 4h         -> qo'lda: faqat H4
    python scanner.py all        -> TO'LIQ HISOBOT: barcha coinlar, filtrsiz
    python scanner.py all 4h     -> to'liq hisobot faqat H4 bo'yicha

Telegram sozlamalari: .env faylida (kompyuterda) yoki GitHub Secrets'da:
    TELEGRAM_TOKEN=...
    TELEGRAM_CHAT_ID=...
"""

import html
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import requests

# ================= SOZLAMALAR =================
BASE_URL = "https://data-api.binance.vision"  # faqat bozor ma'lumoti, API kalit kerak emas

TIMEFRAMES = ["1h", "4h", "1d"]
TF_NAMES = {"1h": "H1", "4h": "H4", "1d": "D1"}
TF_HOURS = {"1h": 1, "4h": 4, "1d": 24}
TF_TRADINGVIEW = {"1h": "60", "4h": "240", "1d": "D"}
PERIOD_MS = {tf: h * 3_600_000 for tf, h in TF_HOURS.items()}

# --- RSI ---
RSI_PERIOD = 14
RSI_LEVEL = 30
CANDLE_LIMIT = 300            # har bir taymfreymdan nechta sham olinadi
MIN_CANDLES = 100             # bundan kam sham bo'lsa, coin o'tkazib yuboriladi
MIN_QUOTE_VOLUME = 1_000_000  # 24 soatlik savdo hajmi kamida 1 mln USDT (0 = barcha coinlar)
REPORT_MIN_CANDLES = 30       # to'liq hisobot: RSI hisoblash uchun eng kam sham

# --- Avtomatik rejim (GitHub) ---
MAX_CATCHUP = 3               # GitHub kechiksa, o'tkazib yuborilgan nechta shamgacha qayta tekshiriladi

# --- Divergensiya ---
PIVOT_LEFT = 5                # chuqur: chapdagi 5 shamdan past bo'lishi kerak
PIVOT_RIGHT = 3               # ... va o'ngdagi 3 shamdan (shuning uchun 3 sham kechikadi)
DIV_MIN_DIST = 5              # ikki chuqur orasida kamida 5 sham
DIV_MAX_DIST = 60             # ... ko'pi bilan 60 sham
DIV_MIN_RSI_DIFF = 2.0        # RSI farqi kamida 2 punkt (shovqinni olib tashlash)
DIV_BULL_ZONE = 35            # 1-chuqurda RSI shundan past bo'lishi kerak
CONFIRM_MAX_BARS = {"1h": 15, "4h": 12, "1d": 10}   # shuncha sham ichida tasdiq kelmasa - bekor
REPORT_RECENT_BARS = {"1h": 5, "4h": 3, "1d": 3}    # hisobotda tasdiqlangan signal shuncha sham "faol"

# --- Umumiy ---
MAX_WORKERS = 8               # bir vaqtda nechta so'rov yuboriladi

# --- Telegram ---
TG_RETRIES = 4                # bitta xabar uchun nechta urinish
TG_GAP = 1.5                  # xabarlar orasidagi tanaffus (soniya) - flood limitiga tushmaslik uchun

# Stablecoinlar, fiat va boshqa coinning "nusxalari" (o'ralgan / staking tokenlar)
EXCLUDE_BASES = {
    "USDC", "FDUSD", "TUSD", "USDP", "DAI", "BUSD", "USDE", "USD1",
    "EUR", "AEUR", "EURI", "XUSD", "BFUSD", "RLUSD", "GBP", "TRY", "BRL",
    "WBTC", "WBETH", "BETH", "BNSOL",
}

# bStocks (tokenlashtirilgan AQSh aksiyalari: AAPLB, NVDAB, MUB...) - ular kripto emas.
# Nomi "B" bilan tugaydi. Shu qoidaga tushib qoladigan HAQIQIY kriptolar shu ro'yxatda:
B_SUFFIX_CRYPTO = {"BNB"}

TASHKENT = timezone(timedelta(hours=5))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(BASE_DIR, "state.json")  # avtomatik rejim: oxirgi tekshirilgan shamlar
# ==============================================


def load_env():
    """Shu papkadagi .env faylidan sozlamalarni o'qiydi (GitHub'da esa Secrets ishlatiladi)."""
    path = os.path.join(BASE_DIR, ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env()
TG_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TG_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

_local = threading.local()


# ============================================================
#  BINANCE
# ============================================================

def get_session():
    """Har bir oqim (thread) uchun alohida ulanish."""
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
    return _local.session


def api_get(path, params=None):
    """Binance'ga so'rov yuboradi. Limitga yaqinlashsa yoki xato kelsa - kutib, qayta urinadi."""
    url = BASE_URL + path
    for attempt in range(5):
        try:
            resp = get_session().get(url, params=params, timeout=20)
        except requests.RequestException:
            time.sleep(3 + attempt * 2)
            continue

        # 429 = limit oshdi, 418 = IP vaqtincha bloklandi
        if resp.status_code in (418, 429):
            wait = int(resp.headers.get("Retry-After", 60))
            print(f"  [!] Binance limiti oshdi, {wait} soniya kutamiz...")
            time.sleep(wait)
            continue

        # 5xx = Binance tomonidagi xato, biroz kutib qayta urinamiz
        if resp.status_code >= 500:
            time.sleep(5)
            continue

        resp.raise_for_status()

        # Daqiqalik limit 6000. Unga yaqinlashsak - dam olamiz.
        used = int(resp.headers.get("X-MBX-USED-WEIGHT-1M", 0))
        if used > 4800:
            print(f"  [!] Limitga yaqinlashdik ({used}/6000), 30 soniya dam olamiz...")
            time.sleep(30)

        return resp.json()

    raise RuntimeError(f"So'rov 5 marta muvaffaqiyatsiz bo'ldi: {path}")


def is_stablecoin(ticker):
    """Narxi ~1 dollar atrofida va 24 soatda deyarli qimirlamagan bo'lsa - stablecoin."""
    last = float(ticker["lastPrice"])
    high = float(ticker["highPrice"])
    low = float(ticker["lowPrice"])
    if last <= 0:
        return True
    return 0.97 <= last <= 1.03 and (high - low) / last < 0.02


def is_bstock(base):
    """Binance bStocks (tokenlashtirilgan aksiya): nomi 'B' bilan tugaydi, masalan AAPLB, NVDAB."""
    return base.endswith("B") and len(base) >= 2 and base not in B_SUFFIX_CRYPTO


def get_symbols(min_volume=MIN_QUOTE_VOLUME):
    """Savdodagi USDT juftliklar ro'yxati (hajm bo'yicha saralangan) va chiqarib tashlangan bStocks."""
    info = api_get("/api/v3/exchangeInfo")
    pairs = set()
    bstocks = []
    for s in info["symbols"]:
        if not (s["status"] == "TRADING"
                and s["quoteAsset"] == "USDT"
                and s.get("isSpotTradingAllowed", False)
                and s["baseAsset"] not in EXCLUDE_BASES):
            continue
        if is_bstock(s["baseAsset"]):
            bstocks.append(s["symbol"])
            continue
        pairs.add(s["symbol"])

    tickers = api_get("/api/v3/ticker/24hr", {"type": "MINI"})
    volume = {}
    for t in tickers:
        if t["symbol"] in pairs and not is_stablecoin(t):
            volume[t["symbol"]] = float(t["quoteVolume"])

    symbols = [s for s, v in volume.items() if v >= min_volume]
    symbols.sort(key=lambda s: volume[s], reverse=True)
    return symbols, sorted(bstocks)


def get_closed_candles(symbol, interval):
    """Shamlarni oladi va oxirgi YOPILMAGAN shamni olib tashlaydi."""
    raw = api_get(
        "/api/v3/klines",
        {"symbol": symbol, "interval": interval, "limit": CANDLE_LIMIT},
    )
    df = pd.DataFrame(
        raw,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "tb_base", "tb_quote", "ignore",
        ],
    )
    df = df[["open_time", "open", "high", "low", "close", "volume", "close_time"]].astype(float)

    now_ms = time.time() * 1000
    df = df[df["close_time"] < now_ms].reset_index(drop=True)
    return df


# ============================================================
#  INDIKATORLAR
# ============================================================

def rsi_parts(close, period=RSI_PERIOD):
    """RSI (Wilder usuli - TradingView'dagi bilan bir xil) va uning o'rtacha o'sish/tushish qismlari."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rsi = 100 - 100 / (1 + avg_gain / avg_loss)
    return rsi.to_numpy(), avg_gain.to_numpy(), avg_loss.to_numpy()


def rsi_at_price(prev_close, price, prev_avg_gain, prev_avg_loss, period=RSI_PERIOD):
    """
    Sham shu narxda yopilganda RSI qancha bo'lardi.
    Shamning LOW narxini bersak - sham ichida RSI tushgan eng past qiymat chiqadi
    (narx qancha past bo'lsa, RSI ham shuncha past - shuning uchun eng past nuqta = low).
    """
    delta = price - prev_close
    gain = max(delta, 0.0)
    loss = max(-delta, 0.0)
    ag = (prev_avg_gain * (period - 1) + gain) / period
    al = (prev_avg_loss * (period - 1) + loss) / period
    if al == 0:
        return 100.0
    return 100 - 100 / (1 + ag / al)


def find_pivots(lows):
    """Chuqurlar: chapdagi PIVOT_LEFT va o'ngdagi PIVOT_RIGHT shamdan past bo'lgan shamlar."""
    pivots = []
    for i in range(PIVOT_LEFT, len(lows) - PIVOT_RIGHT):
        if lows[i] < lows[i - PIVOT_LEFT:i].min() and lows[i] <= lows[i + 1:i + PIVOT_RIGHT + 1].min():
            pivots.append(i)
    return pivots


def rsi_line_clean(rv, p1, p2):
    """Ikki RSI chuqurini tutashtiruvchi chiziq ostiga oradagi RSI tushmasligi kerak."""
    for k in range(p1 + 1, p2):
        line = rv[p1] + (rv[p2] - rv[p1]) * (k - p1) / (p2 - p1)
        if np.isnan(rv[k]) or rv[k] < line - 0.5:
            return False
    return True


def reversal_candle(o, h, l, c, start, end):
    """start..end oralig'ida burilish shami bormi: yutuvchi sham yoki bolg'a."""
    for k in range(max(start, 1), end + 1):
        body = abs(c[k] - o[k])
        rng = h[k] - l[k]
        if rng <= 0:
            continue
        engulf = c[k] > o[k] and c[k - 1] < o[k - 1] and c[k] >= o[k - 1] and o[k] <= c[k - 1]
        lower_wick = min(o[k], c[k]) - l[k]
        hammer = lower_wick >= 2 * body and (c[k] - l[k]) >= 0.6 * rng
        if engulf:
            return "yutuvchi sham"
        if hammer:
            return "bolg'a sham"
    return None


def find_setups(o, h, l, c, rv, interval):
    """
    Bullish divergensiyalarni topadi va holatini aniqlaydi.
    Shart: narx PASTROQ chuqur qildi, RSI esa BALANDROQ chuqur qildi.
    Tasdiq: narx ikki chuqur orasidagi cho'qqidan yuqorida YOPILSA.
    Bekor: narx 2-chuqurdan pastda yopilsa.
    """
    n = len(c)
    last = n - 1
    max_conf = CONFIRM_MAX_BARS[interval]
    recent = REPORT_RECENT_BARS[interval]
    pivots = find_pivots(l)
    setups = []

    for p2 in pivots:
        if last - p2 > max_conf + recent:
            continue  # juda eski
        if np.isnan(rv[p2]):
            continue
        known_at = p2 + PIVOT_RIGHT  # chuqur aynan shu shamda ma'lum bo'ladi

        # 1-chuqurni qidiramiz (eng yaqindagi mos keladigani)
        p1 = None
        for cand in reversed(pivots):
            dist = p2 - cand
            if dist < DIV_MIN_DIST:
                continue
            if dist > DIV_MAX_DIST:
                break
            if np.isnan(rv[cand]):
                continue
            ok = (l[p2] < l[cand]                              # narx pastroq
                  and rv[p2] > rv[cand] + DIV_MIN_RSI_DIFF     # RSI balandroq
                  and rv[cand] < DIV_BULL_ZONE                 # 1-chuqur sotilgan zonada
                  and l[cand + 1:p2].min() > l[p2])            # oraliqda 2-chuqurdan past narx yo'q
            if ok and rsi_line_clean(rv, cand, p2):
                p1 = cand
                break
        if p1 is None:
            continue

        # Tasdiq darajasi: ikki chuqur orasidagi eng baland cho'qqi
        level = h[p1 + 1:p2].max()

        conf_idx = None
        invalid = False
        for k in range(p2 + 1, n):
            if k - p2 > max_conf:
                break
            if c[k] < l[p2]:
                invalid = True
                break
            if c[k] > level:
                conf_idx = k
                break

        if conf_idx is not None:
            alert_idx = max(conf_idx, known_at)
            if last - alert_idx > recent:
                continue
            status = "confirmed"
            pattern_end = conf_idx
        elif invalid or last - p2 > max_conf:
            continue  # bekor bo'lgan yoki muddati o'tgan
        else:
            status = "forming"
            alert_idx = known_at
            pattern_end = last

        setups.append({
            "status": status,
            "alert_idx": alert_idx,
            "bars_ago": last - alert_idx,
            "p1_price": float(l[p1]),
            "p2_price": float(l[p2]),
            "p1_rsi": float(rv[p1]),
            "p2_rsi": float(rv[p2]),
            "level": float(level),
            "candle": reversal_candle(o, h, l, c, p2, pattern_end),
        })

    return setups


# ============================================================
#  SKANER
# ============================================================

def analyze(symbol, interval, min_candles, since_ms):
    """
    Bitta coinni tekshiradi. Xato chiqsa - faqat SHU coin o'tkazib yuboriladi,
    butun skaner to'xtamaydi.
    """
    try:
        return _analyze(symbol, interval, min_candles, since_ms)
    except Exception as e:
        print(f"  [x] {symbol} ({interval}): {type(e).__name__}: {e}")
        return None


def _analyze(symbol, interval, min_candles, since_ms):
    """
    since_ms = shu vaqtdan KEYIN yopilgan shamlardagi hodisalar "yangi" hisoblanadi.
               None bo'lsa - faqat oxirgi yopilgan sham.
    """
    df = get_closed_candles(symbol, interval)

    if len(df) < min_candles:
        return None  # yangi coin, tarix yetarli emas

    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()
    close_ms = (df["open_time"].to_numpy() + PERIOD_MS[interval]).astype(np.int64)
    rv, ag, al = rsi_parts(df["close"])

    n = len(df)
    last = n - 1
    if np.isnan(rv[last]) or np.isnan(rv[last - 1]):
        return None
    if since_ms is None:
        since_ms = int(close_ms[last - 1])

    # --- RSI 30 hodisalari (yangi yopilgan shamlarda) ---
    events = []
    for i in range(max(last - MAX_CATCHUP + 1, 2), n):
        if close_ms[i] <= since_ms:
            continue
        prev, cur = rv[i - 1], rv[i]
        if np.isnan(prev) or np.isnan(cur):
            continue
        low_rsi = rsi_at_price(c[i - 1], l[i], ag[i - 1], al[i - 1])

        if prev >= RSI_LEVEL and cur < RSI_LEVEL:
            kind = "close"     # 30 dan pastda yopildi
        elif prev >= RSI_LEVEL and low_rsi < RSI_LEVEL:
            kind = "wick"      # impuls bilan 30 ni kesdi, lekin tepada yopildi
        elif prev < RSI_LEVEL and cur >= RSI_LEVEL:
            kind = "up"        # 30 dan yuqoriga qaytdi
        else:
            continue

        events.append({
            "symbol": symbol,
            "kind": kind,
            "rsi": round(float(cur), 1),
            "rsi_low": round(float(low_rsi), 1),
            "price": float(c[i]),
            "low": float(l[i]),
            "close_ms": int(close_ms[i]),
            "bars_ago": last - i,
        })

    # --- Divergensiya ---
    setups = find_setups(o, h, l, c, rv, interval)
    for s in setups:
        idx = s["alert_idx"]
        s["new"] = bool(close_ms[idx] > since_ms and idx > last - MAX_CATCHUP)
        s["symbol"] = symbol
        s["price"] = float(c[last])
    new = [s for s in setups if s["new"]]
    divs = new if new else setups[-1:]

    return {
        "symbol": symbol,
        "rsi": round(float(rv[last]), 1),
        "price": float(c[last]),
        "open_time": float(df["open_time"].iloc[-1]),
        "close_ms": int(close_ms[last]),
        "events": events,
        "divs": divs,
    }


def scan_timeframe(symbols, interval, min_candles=MIN_CANDLES, since_ms=None):
    """Bitta taymfreym bo'yicha barcha coinlarni parallel tekshiradi."""
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        results = [r for r in pool.map(lambda s: analyze(s, interval, min_candles, since_ms), symbols) if r]

    events = [e for r in results for e in r["events"]]
    below = sorted([r for r in results if r["rsi"] < RSI_LEVEL], key=lambda r: r["rsi"])
    divs = [d for r in results for d in r["divs"]]
    divs.sort(key=lambda d: (d["status"] != "confirmed", d["bars_ago"]))

    # Eng ko'p uchraydigan oxirgi sham (yangi coinlar chalg'itmasligi uchun)
    candle_open, last_close_ms = None, None
    if results:
        times = [r["open_time"] for r in results]
        common = max(set(times), key=times.count)
        candle_open = datetime.fromtimestamp(common / 1000, TASHKENT)
        last_close_ms = max(r["close_ms"] for r in results)

    return {
        "events": events, "below": below, "divs": divs,
        "total": len(results), "candle_open": candle_open, "last_close_ms": last_close_ms,
    }


# ============================================================
#  XABARLAR
# ============================================================

def candle_label(candle_open, interval):
    if candle_open is None:
        return "?"
    end = candle_open + timedelta(hours=TF_HOURS[interval])
    return f"{candle_open:%d.%m %H:%M} – {end:%d.%m %H:%M}"


def tv_link(symbol, interval):
    link = f"https://www.tradingview.com/chart/?symbol=BINANCE:{symbol}&interval={TF_TRADINGVIEW[interval]}"
    return f'<a href="{html.escape(link)}">{html.escape(symbol)}</a>'


def late_tag(e):
    return f"  ⏱ {e['bars_ago']} sham oldin" if e["bars_ago"] else ""


def event_line(e, interval):
    link = tv_link(e["symbol"], interval)
    if e["kind"] == "wick":
        return f"{link}  RSI min {e['rsi_low']} → yopildi {e['rsi']}  |  {e['price']:.8g}{late_tag(e)}"
    return f"{link}  RSI {e['rsi']}  |  {e['price']:.8g}{late_tag(e)}"


def coin_line(r, interval):
    return f'{tv_link(r["symbol"], interval)}  RSI {r["rsi"]}  |  {r["price"]:.8g}'


def div_lines(d, interval):
    if d["status"] == "confirmed":
        status = "✅ TASDIQLANDI"
        if d["bars_ago"]:
            status += f" ({d['bars_ago']} sham oldin)"
    else:
        status = "👀 shakllanmoqda"

    lines = [
        f"🟢 {status}: {tv_link(d['symbol'], interval)}",
        f"   Narx: {d['p1_price']:.6g} → {d['p2_price']:.6g}  |  RSI: {d['p1_rsi']:.1f} → {d['p2_rsi']:.1f}",
    ]
    candle = f"  |  🕯 {d['candle']}" if d["candle"] else ""
    if d["status"] == "confirmed":
        lines.append(f"   Struktura buzildi: {d['level']:.6g} dan yuqorida yopildi{candle}")
    else:
        lines.append(f"   Tasdiq uchun: {d['level']:.6g} dan yuqorida yopilishi kerak{candle}")
    lines.append(f"   🛑 Bekor: {d['p2_price']:.6g} dan pastda yopilsa  |  hozir {d['price']:.6g}")
    return lines


def event_sections(events, interval):
    """RSI 30 hodisalarini 3 guruhga ajratib, Telegram qatorlarini tuzadi."""
    groups = [
        ("close", f"🔻 <b>RSI {RSI_LEVEL} dan PASTDA yopildi"),
        ("wick", f"📍 <b>Impuls bilan RSI {RSI_LEVEL} ni kesdi, tepada yopildi"),
        ("up", f"🔼 <b>RSI {RSI_LEVEL} dan YUQORIGA qaytdi"),
    ]
    lines = []
    for kind, title in groups:
        items = sorted([e for e in events if e["kind"] == kind],
                       key=lambda e: (e["bars_ago"], e["rsi_low"] if kind == "wick" else e["rsi"]))
        if not items:
            continue
        lines.append(f"{title} ({len(items)} ta):</b>")
        lines += [event_line(e, interval) for e in items]
        lines.append("")
    return lines


def build_message_lines(interval, res):
    """Oddiy/avtomatik rejim: faqat YANGI hodisalar. Hech narsa bo'lmasa - None (xabar yuborilmaydi)."""
    events = res["events"]
    new_divs = [d for d in res["divs"] if d["new"]]
    if not events and not new_divs:
        return None

    name = TF_NAMES[interval]
    lines = [f"<b>📊 {name}</b>", f"🕐 Sham: {candle_label(res['candle_open'], interval)}", ""]
    lines += event_sections(events, interval)

    if new_divs:
        lines.append(f"<b>📐 BULLISH DIVERGENSIYA ({len(new_divs)} ta):</b>")
        for d in new_divs:
            lines += div_lines(d, interval)
        lines.append("")

    lines.append(f"📉 Hozir RSI {RSI_LEVEL} dan pastda: {len(res['below'])} / {res['total']} coin")
    return lines


def build_report_lines(interval, res):
    """To'liq hisobot: shu shamdagi hodisalar, faol divergensiyalar va RSI 30 dan pastdagi BARCHA coinlar."""
    name = TF_NAMES[interval]
    below, divs = res["below"], res["divs"]

    lines = [f"<b>📋 {name} | TO'LIQ HISOBOT</b>", f"🕐 Sham: {candle_label(res['candle_open'], interval)}", ""]
    lines += event_sections(res["events"], interval)

    lines.append(f"<b>📐 Faol bullish divergensiyalar: {len(divs)} ta</b>")
    if divs:
        for d in divs:
            lines += div_lines(d, interval)
    else:
        lines.append("Hozir faol divergensiya yo'q.")

    lines += ["", f"<b>📉 RSI {RSI_LEVEL} dan pastda: {len(below)} / {res['total']} coin</b>"]
    if below:
        lines += [coin_line(r, interval) for r in below]
    else:
        lines.append("Hozir hech bir coin RSI 30 dan pastda emas.")
    return lines


def send_telegram(text):
    """
    Bitta xabar yuboradi. Flood limiti (429) yoki vaqtinchalik xato kelsa - kutib, qayta urinadi.
    Faqat tuzatib bo'lmaydigan xatoda (noto'g'ri token, chat topilmadi) darhol to'xtaydi.
    """
    if not TG_TOKEN or not TG_CHAT_ID:
        print("  [!] TELEGRAM_TOKEN yoki TELEGRAM_CHAT_ID topilmadi (.env yoki GitHub Secrets)")
        return False

    for attempt in range(TG_RETRIES):
        try:
            resp = requests.post(
                f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                data={
                    "chat_id": TG_CHAT_ID,
                    "text": text[:4096],
                    "parse_mode": "HTML",
                    "disable_web_page_preview": "true",
                },
                timeout=20,
            )
        except requests.RequestException as e:
            print(f"  [!] Telegram'ga ulanib bo'lmadi: {e}")
            time.sleep(3 + attempt * 3)
            continue

        if resp.ok:
            return True

        # 429 = juda ko'p xabar yubordik. Telegram qancha kutishni o'zi aytadi.
        if resp.status_code == 429:
            wait = 5
            try:
                wait = int(resp.json()["parameters"]["retry_after"])
            except Exception:
                pass
            print(f"  [!] Telegram limiti: {wait} soniya kutamiz...")
            time.sleep(wait + 1)
            continue

        # 5xx = Telegram tomonidagi vaqtinchalik xato
        if resp.status_code >= 500:
            time.sleep(3 + attempt * 3)
            continue

        # 400/401/403 = token yoki xabar matnida xato, qayta urinish foyda bermaydi
        print(f"  [!] Telegram xatosi: {resp.status_code} {resp.text}")
        return False

    print(f"  [!] Telegram: {TG_RETRIES} urinishdan keyin ham yuborilmadi")
    return False


def send_long(lines):
    """Uzun ro'yxatni Telegram limiti (4096 belgi) bo'yicha bir nechta xabarga bo'lib yuboradi."""
    chunks = []
    chunk = ""
    for line in lines:
        if len(chunk) + len(line) + 1 > 3800:
            chunks.append(chunk)
            chunk = ""
        chunk += line + "\n"
    if chunk.strip():
        chunks.append(chunk)

    ok = True
    for i, part in enumerate(chunks):
        if i:
            time.sleep(TG_GAP)  # flood limitiga tushmaslik uchun tanaffus
        ok = send_telegram(part) and ok
    return ok


# ============================================================
#  HOLAT (avtomatik rejim uchun: qaysi sham allaqachon tekshirilgan)
# ============================================================

def load_state():
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f)


# ============================================================
#  ASOSIY
# ============================================================

def print_summary(name, tf, res, start):
    def names(items):
        return ", ".join(x["symbol"] for x in items) or "-"

    ev = res["events"]
    print(f"--- {name} (sham: {candle_label(res['candle_open'], tf)}, Toshkent) ---")
    print(f"  RSI 30 dan PASTDA yopildi:  {names([e for e in ev if e['kind'] == 'close'])}")
    print(f"  Impuls bilan 30 ni kesdi:   {names([e for e in ev if e['kind'] == 'wick'])}")
    print(f"  RSI 30 dan YUQORIGA qaytdi: {names([e for e in ev if e['kind'] == 'up'])}")
    print(f"  RSI hozir 30 dan pastda:    {len(res['below'])} / {res['total']}")
    print(f"  Divergensiya tasdiqlandi:   {names([d for d in res['divs'] if d['status'] == 'confirmed'])}")
    print(f"  Divergensiya shakllanmoqda: {names([d for d in res['divs'] if d['status'] == 'forming'])}")
    print(f"  ({time.time() - start:.0f} soniyada tekshirildi)")


def run_auto(report=False):
    """
    GitHub uchun: har ishga tushganda faqat OXIRGI TEKSHIRUVDAN KEYIN yopilgan shamlarni ko'radi.
    GitHub kechiksa ham signal yo'qolmaydi (MAX_CATCHUP shamgacha), takrorlanmaydi ham.
    report=True: har yangi yopilgan sham uchun TO'LIQ HISOBOT (barcha coinlar, filtrsiz).
    """
    state = load_state()
    now_ms = int(time.time() * 1000)

    due = []
    for tf in TIMEFRAMES:
        boundary = now_ms // PERIOD_MS[tf] * PERIOD_MS[tf]  # oxirgi yopilgan shamning yopilish vaqti
        done = state.get(tf)
        if done is not None and done >= boundary:
            print(f"{TF_NAMES[tf]}: yangi yopilgan sham yo'q - o'tkazib yuborildi")
            continue
        due.append(tf)

    if not due:
        return

    min_volume = 0 if report else MIN_QUOTE_VOLUME
    min_candles = REPORT_MIN_CANDLES if report else MIN_CANDLES

    print("Avtomatik: TO'LIQ HISOBOT (filtrsiz)" if report else "Avtomatik: faqat yangi hodisalar")
    print("Coinlar ro'yxati olinmoqda...")
    symbols, _ = get_symbols(min_volume)
    print(f"{len(symbols)} ta USDT juftlik tanlandi\n")

    for tf in due:
        start = time.time()
        res = scan_timeframe(symbols, tf, min_candles, state.get(tf))
        print_summary(TF_NAMES[tf], tf, res, start)

        lines = build_report_lines(tf, res) if report else build_message_lines(tf, res)
        sent = True
        if lines:
            sent = send_long(lines)
            print("  Telegram'ga yuborildi." if sent else "  Telegram'ga yuborilmadi - keyingi safar qayta uriniladi.")
        else:
            print("  Yangi hodisa yo'q - xabar yuborilmadi.")

        if sent and res["last_close_ms"]:
            state[tf] = res["last_close_ms"]
            save_state(state)
        print()


def main():
    args = sys.argv[1:]

    if args == ["test"]:
        ok = send_telegram("✅ <b>Binance RSI skaner</b> Telegram'ga ulandi!")
        print("Test xabari yuborildi - Telegram'ni tekshiring." if ok else "Test xabari yuborilmadi.")
        return

    if args and args[0] == "auto":
        run_auto(report="all" in args[1:])
        return

    report = bool(args) and args[0] == "all"
    if report:
        args = args[1:]
    tfs = args or TIMEFRAMES

    for tf in tfs:
        if tf not in TIMEFRAMES:
            print(f"Noma'lum taymfreym: {tf}. Mumkin bo'lganlar: {', '.join(TIMEFRAMES)}, test, auto, all")
            return

    min_volume = 0 if report else MIN_QUOTE_VOLUME
    min_candles = REPORT_MIN_CANDLES if report else MIN_CANDLES

    print("TO'LIQ HISOBOT rejimi (filtrsiz)" if report else "Qo'lda: oxirgi yopilgan sham")
    print("Coinlar ro'yxati olinmoqda...")
    symbols, bstocks = get_symbols(min_volume)
    print(f"{len(symbols)} ta USDT juftlik tanlandi")
    print(f"bStocks (aksiyalar) chiqarib tashlandi: {len(bstocks)} ta\n")

    for tf in tfs:
        start = time.time()
        res = scan_timeframe(symbols, tf, min_candles)
        print_summary(TF_NAMES[tf], tf, res, start)

        lines = build_report_lines(tf, res) if report else build_message_lines(tf, res)
        if lines:
            sent = send_long(lines)
            print("  Telegram'ga yuborildi." if sent else "  Telegram'ga yuborilmadi.")
        else:
            print("  Yangi hodisa yo'q - Telegram'ga xabar yuborilmadi.")
        print()


if __name__ == "__main__":
    main()