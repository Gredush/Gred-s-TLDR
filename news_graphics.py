"""Γραφικά "έκτακτου δελτίου" πάνω σε εικόνα (τίτλοι, lower third, ticker).

Το μοντέλο εικόνας φτιάχνει μόνο τη φωτορεαλιστική σκηνή (χωρίς κείμενο).
Το κείμενο ζωγραφίζεται εδώ με Pillow, ώστε τα ελληνικά να βγαίνουν σωστά.
"""
import io
import os
import unicodedata
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(BASE_DIR, "fonts")

BOLD_CANDIDATES = [
    os.path.join(FONT_DIR, "LiberationSans-Bold.ttf"),
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
REGULAR_CANDIDATES = [
    os.path.join(FONT_DIR, "LiberationSans-Regular.ttf"),
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

NAVY = (8, 24, 68)
NAVY_DARK = (4, 12, 40)
RED = (200, 16, 46)
WHITE = (255, 255, 255)
YELLOW = (255, 214, 10)


@lru_cache(maxsize=64)
def _font(bold: bool, size: int) -> ImageFont.FreeTypeFont:
    for path in (BOLD_CANDIDATES if bold else REGULAR_CANDIDATES):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    print("[news_graphics] ΔΕΝ βρέθηκε γραμματοσειρά με ελληνικά! Βάλε τον φάκελο fonts/.")
    return ImageFont.load_default()


def greek_upper(text: str) -> str:
    """Κεφαλαία χωρίς τόνους (όπως στους τίτλους των ειδήσεων): 'έκτακτο' -> 'ΕΚΤΑΚΤΟ'."""
    decomposed = unicodedata.normalize("NFD", text.upper())
    stripped = "".join(ch for ch in decomposed if ch != "\u0301")  # αφαίρεση τόνου
    return unicodedata.normalize("NFC", stripped)


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_w: int) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if draw.textlength(trial, font=font) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def _fit_text(draw, text, bold, max_w, max_lines, start, min_size):
    """Βρίσκει το μεγαλύτερο μέγεθος γραμματοσειράς που χωράει σε max_lines γραμμές."""
    for size in range(start, min_size - 1, -2):
        font = _font(bold, size)
        lines = _wrap(draw, text, font, max_w)
        if len(lines) <= max_lines and all(draw.textlength(l, font=font) <= max_w for l in lines):
            return font, lines
    font = _font(bold, min_size)
    lines = _wrap(draw, text, font, max_w)[:max_lines]
    if lines:
        last = lines[-1]
        while last and draw.textlength(last + "…", font=font) > max_w:
            last = last[:-1]
        lines[-1] = last.rstrip() + "…"
    return font, lines


def _cover_resize(img: Image.Image) -> Image.Image:
    """Κόβει σε 16:9 (κρατώντας το πάνω μέρος, γιατί κάτω μπαίνουν τα γραφικά) και κάνει resize."""
    img = img.convert("RGB")
    w, h = img.size
    target = W / H
    if w / h > target:  # πολύ πλατιά -> κόψε πλάγια
        new_w = int(h * target)
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))
    else:  # πολύ ψηλή/τετράγωνη -> κόψε πάνω-κάτω με προτίμηση στο πάνω μέρος
        new_h = int(w / target)
        top = int((h - new_h) * 0.25)
        img = img.crop((0, top, w, top + new_h))
    return img.resize((W, H), Image.LANCZOS)


def render_news_frame(
    image_bytes: bytes,
    headline: str,
    subhead: str,
    ticker: list[str],
    time_str: str,
    channel_name: str = "TL;DR NEWS",
) -> bytes:
    base = _cover_resize(Image.open(io.BytesIO(image_bytes))).convert("RGBA")
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    headline = greek_upper(headline.strip())
    subhead = subhead.strip()
    ticker = [greek_upper(t.strip()) for t in ticker if t and t.strip()]

    # --- Γεωμετρία ---
    ticker_h = 54
    sub_h = 46
    head_h = 128
    ticker_y = H - ticker_h
    sub_y = ticker_y - sub_h
    head_y = sub_y - head_h
    margin = 36

    # Ελαφριά σκίαση κάτω για αναγνωσιμότητα
    g_top = max(head_y - 120, 0)
    g_h = H - g_top
    grad = Image.new("RGBA", (W, g_h), (0, 0, 0, 0))
    gd = ImageDraw.Draw(grad)
    for y in range(g_h):
        gd.line([(0, y), (W, y)], fill=(0, 0, 0, int(190 * (y / g_h) ** 1.4)))
    overlay.paste(grad, (0, g_top), grad)

    # --- Κεφαλίδα επάνω: logo καναλιού + ΖΩΝΤΑΝΑ ---
    bug_font = _font(True, 26)
    bug_w = int(d.textlength(channel_name, font=bug_font)) + 36
    d.rectangle([margin, 28, margin + bug_w, 70], fill=NAVY_DARK + (235,))
    d.rectangle([margin, 28, margin + 8, 70], fill=RED + (255,))
    d.text((margin + 22, 49), channel_name, font=bug_font, fill=WHITE, anchor="lm")

    live_font = _font(True, 24)
    live_text = "ΖΩΝΤΑΝΑ"
    live_w = int(d.textlength(live_text, font=live_font)) + 62
    lx1 = W - margin
    lx0 = lx1 - live_w
    d.rectangle([lx0, 28, lx1, 70], fill=RED + (240,))
    d.ellipse([lx0 + 16, 42, lx0 + 32, 58], fill=WHITE + (255,))
    d.text((lx0 + 44, 49), live_text, font=live_font, fill=WHITE, anchor="lm")

    # --- Κόκκινη ετικέτα ΕΚΤΑΚΤΟ ---
    tag_font = _font(True, 28)
    tag_text = "ΕΚΤΑΚΤΟ"
    tag_w = int(d.textlength(tag_text, font=tag_font)) + 44
    tag_h = 40
    d.rectangle([margin, head_y - tag_h, margin + tag_w, head_y], fill=RED + (255,))
    d.text((margin + tag_w / 2, head_y - tag_h / 2), tag_text, font=tag_font, fill=WHITE, anchor="mm")

    # --- Κύριος τίτλος ---
    d.rectangle([margin, head_y, W - margin, head_y + head_h], fill=NAVY + (240,))
    d.rectangle([margin, head_y, margin + 8, head_y + head_h], fill=YELLOW + (255,))
    text_x = margin + 30
    text_w = (W - margin) - text_x - 24
    font, lines = _fit_text(d, headline, True, text_w, 2, 50, 28)
    line_h = font.size * 1.22
    total = line_h * len(lines)
    y = head_y + (head_h - total) / 2
    for line in lines:
        d.text((text_x, y), line, font=font, fill=WHITE)
        y += line_h

    # --- Υπότιτλος ---
    d.rectangle([margin, sub_y, W - margin, sub_y + sub_h], fill=(235, 238, 245, 245))
    if subhead:
        sfont, slines = _fit_text(d, subhead, False, text_w, 1, 26, 18)
        if slines:
            d.text((text_x, sub_y + sub_h / 2), slines[0], font=sfont, fill=NAVY_DARK + (255,), anchor="lm")

    # --- Ticker κάτω ---
    d.rectangle([0, ticker_y, W, H], fill=NAVY_DARK + (250,))
    clock_font = _font(True, 28)
    clock_w = 140
    d.rectangle([0, ticker_y, clock_w, H], fill=RED + (255,))
    d.text((clock_w / 2, ticker_y + ticker_h / 2), time_str, font=clock_font, fill=WHITE, anchor="mm")

    tick_font = _font(True, 22)
    sep = "  ●  "
    avail = W - clock_w - 40
    items = ticker[:]
    tick_text = sep.join(items)
    while len(items) > 1 and d.textlength(tick_text, font=tick_font) > avail:
        items.pop()
        tick_text = sep.join(items)
    if items and d.textlength(tick_text, font=tick_font) > avail:
        tfont, tl = _fit_text(d, tick_text, True, avail, 1, 22, 16)
        tick_text = tl[0] if tl else ""
        tick_font = tfont
    if tick_text:
        d.text((clock_w + 24, ticker_y + ticker_h / 2), tick_text, font=tick_font, fill=WHITE, anchor="lm")

    out = Image.alpha_composite(base, overlay).convert("RGB")
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=92)
    return buf.getvalue()