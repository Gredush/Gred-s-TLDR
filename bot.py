import asyncio
import base64
import io
import json
import os
import time
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, time as dtime
from functools import partial
from typing import Awaitable, Callable
from zoneinfo import ZoneInfo

import aiohttp
import discord
from discord import app_commands
from discord.ext import commands, tasks
from dotenv import load_dotenv
from google import genai
from google.genai import types
from groq import AsyncGroq

from news_graphics import render_news_frame

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")  # προαιρετικό
AUTO_TLDR_CHANNEL_ID = os.getenv("AUTO_TLDR_CHANNEL_ID")

# Εικόνα (προαιρετικά: χωρίς αυτά χρησιμοποιείται το Pollinations)
CF_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID")
CF_API_TOKEN = os.getenv("CF_API_TOKEN")
CF_IMAGE_MODEL = os.getenv("CF_IMAGE_MODEL", "@cf/black-forest-labs/flux-1-schnell")
CF_IMAGE_STEPS = int(os.getenv("CF_IMAGE_STEPS", "6"))  # schnell: 1-8
IMAGE_ENABLED = os.getenv("IMAGE_ENABLED", "true").lower() == "true"
IMAGE_TIMEOUT = 45          # ανά πάροχο εικόνας
IMAGE_TOTAL_TIMEOUT = 100   # σύνολο για σχέδιο + εικόνα + γραφικά


def env_list(name: str, default: str) -> list[str]:
    return [x.strip() for x in os.getenv(name, default).split(",") if x.strip()]


# ------------------------------------------------------------------
# Ρυθμίσεις μοντέλων (άλλαξέ τα από env χωρίς να πειράξεις κώδικα)
# ------------------------------------------------------------------
GEMINI_MODELS = env_list(
    "GEMINI_MODELS",
    "gemini-3.5-flash,gemini-3.1-flash-lite,gemini-2.5-flash-lite",
)
GROQ_MODELS = env_list("GROQ_MODELS", "llama-3.3-70b-versatile,llama-3.1-8b-instant")
CEREBRAS_MODEL = os.getenv("CEREBRAS_MODEL", "gpt-oss-120b")  # έλεγξε το ID στο dashboard

# Μέγιστοι χαρακτήρες log ανά πάροχο (τα ελληνικά τρώνε πολλά tokens).
GEMINI_MAX_CHARS = 60_000
CEREBRAS_MAX_CHARS = 30_000
GROQ_MAX_CHARS = {          # με βάση τα free TPM (12k / 6k) + system prompt + output
    "llama-3.3-70b-versatile": 14_000,
    "llama-3.1-8b-instant": 6_500,
}
GROQ_DEFAULT_MAX_CHARS = 6_500

PER_CALL_TIMEOUT = 40   # δευτερόλεπτα ανά κλήση
TOTAL_TIMEOUT = 110     # συνολικό όριο για όλη την αλυσίδα

# ------------------------------------------------------------------
# Catch-up TL;DR: μόνο όταν η δραστηριότητα της ΤΕΛΕΥΤΑΙΑΣ ΩΡΑΣ είναι αισθητά πάνω από το συνηθισμένο
# ------------------------------------------------------------------
CATCHUP_ENABLED = os.getenv("CATCHUP_ENABLED", "true").lower() == "true"
# "Έξαρση" = χαρακτήρες τελευταίας ώρας >= FACTOR x τυπική ενεργή ώρα του καναλιού (τελευταίο 24ωρο)
CATCHUP_SURGE_FACTOR = float(os.getenv("CATCHUP_SURGE_FACTOR", "3"))
# ...και τουλάχιστον τόσοι χαρακτήρες την τελευταία ώρα (αποφεύγει "έξαρση" σε νεκρά κανάλια)
CATCHUP_MIN_RECENT_CHARS = int(os.getenv("CATCHUP_MIN_RECENT_CHARS", "6000"))
# Ελάχιστοι νέοι χαρακτήρες από την τελευταία σύνοψη, ώστε να υπάρχει κάτι άξιο σύνοψης
CATCHUP_MIN_NEW_CHARS = int(os.getenv("CATCHUP_MIN_NEW_CHARS", "5000"))
CATCHUP_BASELINE_HOURS = int(os.getenv("CATCHUP_BASELINE_HOURS", "24"))
CATCHUP_BASELINE_FLOOR = int(os.getenv("CATCHUP_BASELINE_FLOOR", "1500"))  # χαρ./ώρα (κατώτατο baseline)
CATCHUP_MIN_INTERVAL = int(os.getenv("CATCHUP_MIN_INTERVAL", "900"))       # ελάχιστο κενό ανά κανάλι (δευτ.)
CATCHUP_WITH_IMAGE = os.getenv("CATCHUP_WITH_IMAGE", "false").lower() == "true"
# Προαιρετικά: μόνο συγκεκριμένα κανάλια (IDs χωρισμένα με κόμμα). Κενό = όλα.
CATCHUP_CHANNEL_IDS = {int(x) for x in env_list("CATCHUP_CHANNEL_IDS", "") if x.isdigit()}

groq_client = (
    AsyncGroq(api_key=GROQ_API_KEY, max_retries=0, timeout=PER_CALL_TIMEOUT)
    if GROQ_API_KEY else None
)
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)
GUILD_ID = None

COOLDOWN_SECONDS = 1800
last_used: dict[int, float] = {}
in_progress: set[int] = set()

# Κατάσταση catch-up ανά κανάλι
catchup_chars: dict[int, int] = {}          # χαρακτήρες log από την τελευταία σύνοψη
catchup_since: dict[int, datetime] = {}     # από πότε μετράμε
catchup_last_run: dict[int, float] = {}     # monotonic χρόνος τελευταίου catch-up
catchup_retry_at: dict[int, float] = {}     # επόμενη δοκιμή μετά από αποτυχία
catchup_disabled: set[int] = set()          # κανάλια χωρίς δικαιώματα

# Ιστορικό δραστηριότητας: ανά κανάλι, χαρακτήρες ανά λεπτό (για να βρίσκουμε εξάρσεις)
activity_buckets: dict[int, dict[int, int]] = {}
activity_ready: set[int] = set()            # κανάλια με φορτωμένο ιστορικό 24ώρου
activity_bootstrapping: set[int] = set()
activity_last_check: dict[int, float] = {}
bootstrap_sem = asyncio.Semaphore(2)        # όχι πολλά history fetch μαζί
SUMMARY_PREFIXES = ("**TL;DR", "🤖 **Αυτόματο TL;DR", "⚡ **Catch-up TL;DR")
background_tasks: set = set()

ATHENS_TZ = ZoneInfo("Europe/Athens")
AUTO_TLDR_HOURS = 4
AUTO_TLDR_TIMES = [
    dtime(hour=(9 + AUTO_TLDR_HOURS * i) % 24, minute=0, tzinfo=ATHENS_TZ)
    for i in range(24 // AUTO_TLDR_HOURS)
]

# Κανάλι με "πικάντικο" ύφος
SPICY_CHANNEL_ID = 785612058360021033


# ------------------------------------------------------------------
# Prompts
# ------------------------------------------------------------------
BASE_PROMPT = """\
Είσαι ένας ακριβής και αντικειμενικός αναλυτής συνομιλιών Discord σε community chat.
Η αποστολή σου είναι να διαβάζεις ιστορικά συνομιλιών (Ελληνικά, Greeklish, Αγγλικά, slang) και να παράγεις μια καθαρή, εύκολα κατανοητή σύνοψη (TL;DR) αποκλειστικά στα Ελληνικά, ώστε κάποιος που μόλις μπήκε να καταλάβει γρήγορα τι συζητήθηκε.

### ΟΔΗΓΙΕΣ ΣΚΕΨΗΣ (ΜΗΝ ΤΙΣ ΕΜΦΑΝΙΖΕΙΣ)
Πριν γράψεις την τελική σύνοψη, σκέψου εσωτερικά:
1. Εντόπισε τα βασικά θέματα που συζητήθηκαν.
2. Για κάθε θέμα σημείωσε σύντομα πώς ξεκίνησε και πώς εξελίχθηκε.
3. Δες αν υπήρξαν αξιοσημείωτες τοποθετήσεις, αστεία ή έντονες συζητήσεις.
4. Αγνόησε πλήρως: εικόνες, GIFs, emojis, bot messages, reactions και links που δεν συνεισφέρουν στη συζήτηση.
5. Μην προσθέτεις τίποτα που δεν υπάρχει ρητά στα μηνύματα. Αν κάτι δεν είναι σαφές, μην το αναφέρεις.

### ΚΑΝΟΝΕΣ ΕΞΟΔΟΥ
- Γράψε μόνο στα Ελληνικά, με σωστή ορθογραφία και φυσική ροή.
- Τόνος: ουδέτερος, καθαρός και εύκολα αναγνώσιμος (σαν να εξηγείς σε κάποιον τι έγινε όσο έλειπε).
- Μην κρίνεις, μην σχολιάζεις χαρακτήρες και μην βγάζεις συμπεράσματα.
- Μην εφευρίσκεις ούτε συνδέεις άσχετα θέματα μεταξύ τους.

### ΔΟΜΗ ΕΞΟΔΟΥ (Χρησιμοποίησε αυτή)
📌 **Τι συζητήθηκε**
- **[Θέμα 1]:** Σύντομη περιγραφή του τι ειπώθηκε και πώς κύλησε η κουβέντα.
- **[Θέμα 2]:** ...
- **[Θέμα 3]:** ...
(Αν υπήρχαν πολλά μικρά/περαστικά θέματα, ομαδοποίησέ τα σε μία γραμμή τύπου «Διάφορα μικρά θέματα: ...»)

💡 **Αξιοσημείωτα**
- (Μόνο αν υπήρξε κάτι ιδιαίτερο: έντονη συζήτηση, αστείο που κράτησε, σημαντική τοποθέτηση κλπ. Αν δεν υπάρχει τίποτα ξεχωριστό → παράλειψε εντελώς αυτή την ενότητα)

### ΠΕΡΙΟΡΙΣΜΟΣ ΜΗΚΟΥΣ
Κράτα τη σύνοψη γύρω στις 180–200 λέξεις και ΠΟΤΕ πάνω από 1700 χαρακτήρες συνολικά (πρέπει να χωράει σε ένα μήνυμα Discord).
"""

SPICY_ADDON = """
### ΕΙΔΙΚΗ ΠΡΟΣΘΗΚΗ ΥΦΟΥΣ (ΠΟΛΥ ΠΟΛΥ ΔΙΑΚΡΙΤΙΚΗ)
Σε αυτό το κανάλι ο τόνος είναι ελάχιστα λιγότερο ουδέτερος: ένα ξερό, σχεδόν αδιόρατο «ανασηκωμένο φρύδι». Η δομή και ο όγκος της σύνοψης μένουν ΑΚΡΙΒΩΣ ίδια. Η αλλαγή είναι μικρή και εύκολο να περάσει απαρατήρητη.
- Πρόσθεσε το πολύ 1–2 υπαινιγμούς συνολικά, μία πρόταση ο καθένας, είτε μέσα σε ένα θέμα είτε στα «Αξιοσημείωτα».
- Κάθε υπαινιγμός πρέπει να στηρίζεται σε κάτι που φαίνεται ΠΡΑΓΜΑΤΙΚΑ στο log: ποιος απάντησε ή δεν απάντησε σε μια ερώτηση, ποιος εξαφανίστηκε ξαφνικά, ένα θέμα που κόπηκε απότομα, μια σύμπτωση στη χρονική σειρά, ένα σχόλιο που έμεινε χωρίς συνέχεια.
- Διατύπωσέ τον ως παρατήρηση ή ανοιχτό ερώτημα, ποτέ ως βεβαιότητα. Π.χ. «Το ερώτημα του Χ έμεινε τελικά αναπάντητο… ενδιαφέρον.» ή «Δεν είναι σαφές τι ακριβώς έγινε με το θέμα Υ, αλλά η κουβέντα άλλαξε απότομα.»
- Μην εξηγείς τον υπαινιγμό και μην αναφέρεις ότι σου ζητήθηκε ύφος ή ίντριγκα.
- ΑΠΑΓΟΡΕΥΕΤΑΙ: υπαινιγμοί για σεξουαλικότητα, υγεία, οικογένεια, οικονομικά, θρησκεία, πολιτικές πεποιθήσεις ή προσωπικές σχέσεις. Επίσης απαγορεύονται προσβολές, χλευασμός και αναφορές σε άτομα που δεν μιλούν στο log.
- Αν δεν υπάρχει κάτι πρόσφορο, μη βάλεις τίποτα. Η σύνοψη μένει κανονική.
"""

NEWS_PLAN_SYSTEM = """\
You create a satirical "breaking news" TV frame from a Greek-language summary of a Discord chat. Return ONLY a JSON object (no markdown fences, no commentary) with exactly these keys:

{
  "image_prompt": "...",
  "headline": "...",
  "subhead": "...",
  "ticker": ["...", "...", "..."]
}

image_prompt (English, 50-90 words): a photorealistic still frame of a live TV news studio. A female news anchor (generic, fictional, non-recognisable) sits at a sleek modern studio desk, medium shot, looking straight into the camera with a serious, professional expression. Behind her, a large video wall shows an exaggerated, absurd, photographic visual of the chat's main topic. Describe the lighting, the camera and the video wall content concretely. The image must contain NO text, letters, numbers, captions, tickers or logos anywhere (the news graphics are added later by software). Keep the bottom fifth of the frame simple (desk surface).

headline (Greek, max 70 characters): the breaking-news headline about the most prominent topic discussed. Dead-serious, sensational news tone applied to a trivial or funny subject. Strictly based on what was actually discussed.

subhead (Greek, max 90 characters): a one-line follow-up in the same deadpan news style.

ticker (array of 1-3 Greek strings, each max 40 characters): other topics from the chat as short news-ticker headlines. Use fewer items if there were fewer topics.

Rules for all Greek text:
- Correct Greek with normal accents and normal capitalisation (the software handles capitals).
- Base everything on topics really present in the summary. Do not invent events or facts.
- Never include usernames or real people's names. Refer to "το κανάλι", "μέλη", "η παρέα" instead.
- Gentle satire: laugh at the situation, never at individuals. No insults.
- Avoid sexual content, hate, real tragedies and sensitive topics (religion, politics, health, minors). If the chat covers those, write about a harmless aspect, e.g. an intense discussion in the channel.
"""

IMAGE_STYLE_SUFFIX = (
    " Photorealistic still frame from a live television news broadcast, "
    "professional broadcast camera, soft studio lighting, 16:9 widescreen framing, "
    "sharp focus, highly detailed, no text, no letters, no numbers, no logos, no watermark."
)


# ------------------------------------------------------------------
# Βοηθητικά
# ------------------------------------------------------------------
def fit_to_discord_limit(header: str, text: str, max_limit: int = 1980) -> str:
    full_text = header + text
    if len(full_text) <= max_limit:
        return full_text

    cutoff_note = "\n\n*(Η σύνοψη κόπηκε λόγω ορίου χαρακτήρων)*"
    allowed = max_limit - len(header) - len(cutoff_note)
    trimmed = text[:allowed]
    # κόψε σε αλλαγή γραμμής αν υπάρχει αρκετά κοντά, αλλιώς σε κενό
    for sep in ("\n", " "):
        idx = trimmed.rfind(sep)
        if idx > allowed * 0.6:
            trimmed = trimmed[:idx]
            break
    return header + trimmed + cutoff_note


def build_log(lines: list[str], max_chars: int) -> str:
    """Κρατά τις πιο πρόσφατες γραμμές που χωράνε στο όριο (χρονολογική σειρά)."""
    out, total = [], 0
    for line in reversed(lines):
        if total + len(line) + 1 > max_chars:
            break
        out.append(line)
        total += len(line) + 1
    if not out and lines:
        return lines[-1][-max_chars:]
    return "\n".join(reversed(out))


# ------------------------------------------------------------------
# Πάροχοι AI (κείμενο)
# ------------------------------------------------------------------
@dataclass
class Provider:
    key: str
    max_chars: int
    call: Callable[[str, str], Awaitable[str]]


async def _call_gemini(model: str, system_prompt: str, chat_log: str) -> str:
    resp = await gemini_client.aio.models.generate_content(
        model=model,
        contents=f"Ιστορικό Συνομιλίας:\n{chat_log}",
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.4,
            max_output_tokens=4096,  # περιθώριο και για εσωτερικό "thinking"
        ),
    )
    return resp.text or ""


async def _call_groq(model: str, system_prompt: str, chat_log: str) -> str:
    resp = await groq_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Ιστορικό Συνομιλίας:\n{chat_log}"},
        ],
        temperature=0.3,
        max_tokens=900,  # κρατά χαμηλά το TPM
    )
    return resp.choices[0].message.content or ""


async def _call_cerebras(model: str, system_prompt: str, chat_log: str) -> str:
    url = "https://api.cerebras.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {CEREBRAS_API_KEY}"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Ιστορικό Συνομιλίας:\n{chat_log}"},
        ],
        "temperature": 0.3,
        "max_tokens": 1500,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(
            url, json=payload, headers=headers,
            timeout=aiohttp.ClientTimeout(total=PER_CALL_TIMEOUT),
        ) as resp:
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status}: {(await resp.text())[:200]}")
            data = await resp.json()
            return data["choices"][0]["message"]["content"] or ""


def build_providers() -> list[Provider]:
    providers: list[Provider] = []
    if gemini_client:  # μεγάλο όριο tokens -> πρώτο
        for m in GEMINI_MODELS:
            providers.append(Provider(f"gemini:{m}", GEMINI_MAX_CHARS, partial(_call_gemini, m)))
    if CEREBRAS_API_KEY:
        providers.append(Provider(f"cerebras:{CEREBRAS_MODEL}", CEREBRAS_MAX_CHARS,
                                  partial(_call_cerebras, CEREBRAS_MODEL)))
    if groq_client:  # γρήγορο αλλά με μικρό TPM -> τελευταίο, με κομμένο log
        for m in GROQ_MODELS:
            providers.append(Provider(f"groq:{m}", GROQ_MAX_CHARS.get(m, GROQ_DEFAULT_MAX_CHARS),
                                      partial(_call_groq, m)))
    return providers


PROVIDERS = build_providers()
blocked_until: dict[str, float] = {}


def block_provider(key: str, err: Exception | str) -> None:
    msg = str(err).lower()
    if any(s in msg for s in ("404", "not found", "decommission", "does not exist", "not supported")):
        secs = 6 * 3600
    elif any(s in msg for s in ("per day", "daily", "tpd", "rpd")):
        secs = 3600
    elif any(s in msg for s in ("429", "rate", "quota", "resource_exhausted", "413", "too large")):
        secs = 120
    else:
        secs = 30
    blocked_until[key] = time.monotonic() + secs


class SummaryError(Exception):
    pass


async def generate_summary(system_prompt: str, lines: list[str]) -> str:
    deadline = time.monotonic() + TOTAL_TIMEOUT
    failures = []

    for p in PROVIDERS:
        remaining = deadline - time.monotonic()
        if remaining < 8:
            break
        if time.monotonic() < blocked_until.get(p.key, 0):
            continue
        try:
            text = await asyncio.wait_for(
                p.call(system_prompt, build_log(lines, p.max_chars)),
                timeout=min(PER_CALL_TIMEOUT, remaining),
            )
            if text and text.strip():
                return text.strip()
            failures.append(f"{p.key}: κενή απάντηση")
            block_provider(p.key, "empty")
        except asyncio.TimeoutError:
            failures.append(f"{p.key}: timeout")
            block_provider(p.key, "timeout")
        except Exception as e:
            failures.append(f"{p.key}: {str(e)[:150]}")
            block_provider(p.key, e)

    print("[Summary failures]\n  " + "\n  ".join(failures or ["(όλοι σε cooldown ή δεν υπάρχουν keys)"]))
    raise SummaryError("Όλες οι υπηρεσίες AI είναι προσωρινά απασχολημένες. Δοκίμασε ξανά σε λίγα λεπτά.")


def format_message_line(message: discord.Message) -> str | None:
    """Η ΙΔΙΑ μορφή που μπαίνει στο prompt (χρησιμοποιείται και για το μέτρημα χαρακτήρων)."""
    content = message.clean_content.strip()
    if message.author.bot or not content:
        return None
    ts = message.created_at.astimezone(ATHENS_TZ).strftime("%H:%M")
    return f"[{ts}] {message.author.display_name}: {content[:600]}"


def reset_catchup(channel_id: int) -> None:
    catchup_chars.pop(channel_id, None)
    catchup_since.pop(channel_id, None)


async def fetch_and_generate_tldr(
    channel: discord.TextChannel, hours: int | None = None, after: datetime | None = None
) -> str | None:
    cutoff_time = after if after else datetime.now(timezone.utc) - timedelta(hours=hours)
    lines: list[str] = []

    # Από το πιο πρόσφατο προς τα πίσω, ώστε αν κοπεί κάτι να είναι τα παλιότερα.
    async for message in channel.history(after=cutoff_time, limit=1000, oldest_first=False):
        line = format_message_line(message)
        if line:
            lines.append(line)

    if not lines:
        return None

    lines.reverse()  # παλιό -> νέο

    system_prompt = BASE_PROMPT
    if channel.id == SPICY_CHANNEL_ID:
        system_prompt += SPICY_ADDON

    return await generate_summary(system_prompt, lines)


# ------------------------------------------------------------------
# Εικόνα: φωτορεαλιστικό στούντιο ειδήσεων + γραφικά με τίτλους
# ------------------------------------------------------------------
@dataclass
class NewsPlan:
    image_prompt: str
    headline: str
    subhead: str
    ticker: list[str]


def parse_news_plan(raw: str) -> NewsPlan:
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("Δεν βρέθηκε JSON")
    data = json.loads(raw[start:end + 1])

    def clean(value, limit):
        return " ".join(str(value or "").split())[:limit]

    image_prompt = clean(data.get("image_prompt"), 900).replace('"', "")
    headline = clean(data.get("headline"), 100)
    subhead = clean(data.get("subhead"), 140)
    ticker = [clean(t, 60) for t in (data.get("ticker") or []) if str(t).strip()][:4]
    if not image_prompt or not headline:
        raise ValueError("Λείπουν πεδία (image_prompt/headline)")
    return NewsPlan(image_prompt, headline, subhead, ticker)


async def build_news_plan(summary: str) -> NewsPlan:
    last_err: Exception | None = None
    for _ in range(2):  # μία επανάληψη αν το μοντέλο δεν επιστρέψει έγκυρο JSON
        raw = await generate_summary(NEWS_PLAN_SYSTEM, [summary])
        try:
            return parse_news_plan(raw)
        except Exception as e:
            last_err = e
            print(f"[News plan parse error] {e} | {raw[:200]!r}")
    raise ValueError(f"Αποτυχία σχεδίου δελτίου: {last_err}")


async def _cf_image(prompt: str) -> bytes:
    url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/{CF_IMAGE_MODEL}"
    headers = {"Authorization": f"Bearer {CF_API_TOKEN}"}
    payload = {"prompt": prompt, "steps": CF_IMAGE_STEPS}
    async with aiohttp.ClientSession() as session:
        async with session.post(
            url, json=payload, headers=headers,
            timeout=aiohttp.ClientTimeout(total=IMAGE_TIMEOUT),
        ) as resp:
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status}: {(await resp.text())[:200]}")
            if resp.content_type.startswith("image/"):
                return await resp.read()
            data = await resp.json()
            b64 = (data.get("result") or {}).get("image")
            if not b64:
                raise RuntimeError("Η απάντηση δεν περιείχε εικόνα")
            return base64.b64decode(b64)


async def _pollinations_image(prompt: str) -> bytes:
    url = "https://image.pollinations.ai/prompt/" + urllib.parse.quote(prompt)
    params = {
        "model": "flux", "width": 1024, "height": 576,
        "nologo": "true", "seed": int(time.time()),
    }
    async with aiohttp.ClientSession() as session:
        async with session.get(
            url, params=params,
            timeout=aiohttp.ClientTimeout(total=IMAGE_TIMEOUT),
        ) as resp:
            if resp.status != 200 or not resp.content_type.startswith("image/"):
                raise RuntimeError(f"HTTP {resp.status} ({resp.content_type})")
            return await resp.read()


async def generate_image(summary: str) -> bytes | None:
    """Επιστρέφει bytes εικόνας δελτίου (με τίτλους) ή None. Δεν πετάει ποτέ exception."""
    if not IMAGE_ENABLED:
        return None

    async def _run() -> bytes | None:
        plan = await build_news_plan(summary)
        prompt = plan.image_prompt + IMAGE_STYLE_SUFFIX
        print(f"[Image prompt] {prompt}")
        print(f"[News] {plan.headline} | {plan.subhead} | {plan.ticker}")

        backends = []
        if CF_ACCOUNT_ID and CF_API_TOKEN:
            backends.append(("cloudflare", _cf_image))
        backends.append(("pollinations", _pollinations_image))

        for name, fn in backends:
            try:
                raw = await asyncio.wait_for(fn(prompt), timeout=IMAGE_TIMEOUT)
            except Exception as e:
                print(f"[Image {name} failed] {str(e)[:200]}")
                continue
            try:
                now_str = datetime.now(ATHENS_TZ).strftime("%H:%M")
                return await asyncio.to_thread(
                    render_news_frame, raw, plan.headline, plan.subhead, plan.ticker, now_str
                )
            except Exception as e:
                print(f"[News overlay failed] {e!r}")
                return raw  # καλύτερα η εικόνα χωρίς τίτλους παρά τίποτα
        return None

    try:
        return await asyncio.wait_for(_run(), timeout=IMAGE_TOTAL_TIMEOUT)
    except Exception as e:
        print(f"[Image skipped] {str(e)[:200]}")
        return None


async def make_tldr_payload(
    channel: discord.TextChannel, hours: int | None, header: str,
    after: datetime | None = None, with_image: bool = True,
):
    """Επιστρέφει (κείμενο, discord.File | None) ή None αν δεν υπάρχουν μηνύματα."""
    summary = await fetch_and_generate_tldr(channel, hours=hours, after=after)
    if not summary:
        return None
    text = fit_to_discord_limit(header, summary)
    image = await generate_image(summary) if with_image else None
    file = discord.File(io.BytesIO(image), filename="tldr.jpg") if image else None
    return text, file


# ------------------------------------------------------------------
# Discord
# ------------------------------------------------------------------
@bot.event
async def on_ready():
    print(f"Logged in as {bot.user.name} (ID: {bot.user.id})")
    print(f"Providers: {[p.key for p in PROVIDERS]}")
    print(f"Catch-up: enabled={CATCHUP_ENABLED}, surge>={CATCHUP_SURGE_FACTOR}x, "
          f"min_recent={CATCHUP_MIN_RECENT_CHARS}, channels={sorted(CATCHUP_CHANNEL_IDS) or 'ALL'}")
    print(f"Image: enabled={IMAGE_ENABLED}, cloudflare={'yes' if CF_ACCOUNT_ID and CF_API_TOKEN else 'no'}")
    try:
        if GUILD_ID:
            bot.tree.copy_global_to(guild=GUILD_ID)
            synced = await bot.tree.sync(guild=GUILD_ID)
            print(f"Synced {len(synced)} command(s) to Guild {GUILD_ID.id}")
        else:
            synced = await bot.tree.sync()
            print(f"Synced {len(synced)} command(s) globally.")
    except Exception as e:
        print(f"Sync error: {e}")

    if not auto_tldr_task.is_running():
        auto_tldr_task.start()


async def run_catchup(channel: discord.TextChannel):
    cid = channel.id
    chars = catchup_chars.pop(cid, 0)
    since = catchup_since.pop(cid, None)
    success = False
    try:
        if since is None:
            success = True
            return
        start = since.astimezone(ATHENS_TZ).strftime("%H:%M")
        header = f"⚡ **Catch-up TL;DR** (έντονη δραστηριότητα από τις {start}) 📝\n\n"
        payload = await make_tldr_payload(
            channel, None, header, after=since, with_image=CATCHUP_WITH_IMAGE
        )
        if payload:
            text, file = payload
            try:
                if file:
                    await channel.send(text, file=file)
                else:
                    await channel.send(text)
            except discord.Forbidden:
                raise
            except Exception as e:
                print(f"[Catch-up send with image failed] {e}")
                if file:
                    await channel.send(text)
        success = True
        catchup_last_run[cid] = time.monotonic()
    except discord.Forbidden:
        catchup_disabled.add(cid)
        success = True  # δεν ξαναδοκιμάζουμε σε κανάλι χωρίς δικαιώματα
        print(f"[Catch-up] Χωρίς δικαιώματα στο κανάλι {cid}, απενεργοποιήθηκε.")
    except Exception as e:
        print(f"[Catch-up Error] {e!r}")
    finally:
        if not success and since is not None:
            # επαναφορά μετρητή (συν ό,τι ήρθε στο μεταξύ) και νέα δοκιμή σε 3 λεπτά
            catchup_chars[cid] = catchup_chars.get(cid, 0) + chars
            catchup_since[cid] = since
            catchup_retry_at[cid] = time.monotonic() + 180
        in_progress.discard(cid)


def surge_status(cid: int, now_ts: float) -> tuple[bool, int, float]:
    """(έξαρση;, χαρακτήρες τελευταίας ώρας, τυπική ενεργή ώρα)."""
    buckets = activity_buckets.get(cid) or {}
    now_min = int(now_ts // 60)

    oldest = now_min - CATCHUP_BASELINE_HOURS * 60 + 1
    for m in [m for m in buckets if m < oldest]:
        del buckets[m]  # καθαρισμός παλιών

    hours: dict[int, int] = {}
    for m, c in buckets.items():
        h = max(0, (now_min - m) // 60)   # 0 = τελευταία ώρα, 1 = η προηγούμενη, ...
        hours[h] = hours.get(h, 0) + c

    recent = hours.get(0, 0)
    past = [v for h, v in hours.items() if h >= 1 and v > 0]   # μόνο ώρες με δραστηριότητα
    typical = sum(past) / len(past) if len(past) >= 2 else 0.0
    baseline = max(typical, float(CATCHUP_BASELINE_FLOOR))

    surge = recent >= CATCHUP_MIN_RECENT_CHARS and recent >= CATCHUP_SURGE_FACTOR * baseline
    return surge, recent, baseline


async def bootstrap_activity(channel: discord.TextChannel, before: datetime):
    """Φορτώνει μία φορά το ιστορικό 24ώρου, ώστε το baseline και ο μετρητής να επιβιώνουν από restart."""
    cid = channel.id
    window_start = before - timedelta(hours=CATCHUP_BASELINE_HOURS)
    buckets = activity_buckets.setdefault(cid, {})
    entries: list[tuple[datetime, int]] = []
    last_summary_at: datetime | None = None
    try:
        async with bootstrap_sem:
            async for m in channel.history(after=window_start, before=before, limit=3000, oldest_first=False):
                if (last_summary_at is None and bot.user is not None
                        and m.author.id == bot.user.id and m.content.startswith(SUMMARY_PREFIXES)):
                    last_summary_at = m.created_at
                line = format_message_line(m)
                if not line:
                    continue
                n = len(line) + 1
                minute = int(m.created_at.timestamp() // 60)
                buckets[minute] = buckets.get(minute, 0) + n
                entries.append((m.created_at, n))
    except discord.Forbidden:
        catchup_disabled.add(cid)
        print(f"[Catch-up] Χωρίς πρόσβαση στο ιστορικό του καναλιού {cid}, απενεργοποιήθηκε.")
    except Exception as e:
        print(f"[Catch-up bootstrap Error] {e!r}")
    finally:
        since = last_summary_at or window_start
        acc = sum(n for t, n in entries if t > since)
        if acc > 0:
            catchup_chars[cid] = catchup_chars.get(cid, 0) + acc
            first = catchup_since.get(cid)
            catchup_since[cid] = since if first is None else min(first, since)
        activity_ready.add(cid)
        activity_bootstrapping.discard(cid)


@bot.event
async def on_message(message: discord.Message):
    await bot.process_commands(message)

    if not CATCHUP_ENABLED or message.guild is None:
        return
    channel = message.channel
    if not isinstance(channel, discord.TextChannel):
        return
    cid = channel.id
    if cid in catchup_disabled or (CATCHUP_CHANNEL_IDS and cid not in CATCHUP_CHANNEL_IDS):
        return

    line = format_message_line(message)
    if not line:
        return

    n = len(line) + 1
    minute = int(message.created_at.timestamp() // 60)
    buckets = activity_buckets.setdefault(cid, {})
    buckets[minute] = buckets.get(minute, 0) + n
    if cid not in catchup_since:
        catchup_since[cid] = message.created_at - timedelta(seconds=1)
    catchup_chars[cid] = catchup_chars.get(cid, 0) + n

    # Πρώτη φορά για το κανάλι μετά από εκκίνηση: φόρτωσε ιστορικό 24ώρου στο παρασκήνιο
    if cid not in activity_ready:
        if cid not in activity_bootstrapping:
            activity_bootstrapping.add(cid)
            t = asyncio.create_task(bootstrap_activity(channel, message.created_at))
            background_tasks.add(t)
            t.add_done_callback(background_tasks.discard)
        return

    if catchup_chars[cid] < CATCHUP_MIN_NEW_CHARS or cid in in_progress:
        return
    now = time.monotonic()
    if now < catchup_retry_at.get(cid, 0):
        return
    last = catchup_last_run.get(cid)
    if last is not None and now - last < CATCHUP_MIN_INTERVAL:
        return
    if now - activity_last_check.get(cid, -1e9) < 20:  # έλεγχος έξαρσης το πολύ κάθε 20"
        return
    activity_last_check[cid] = now

    surge, recent, baseline = surge_status(cid, time.time())
    if not surge:
        return
    print(f"[Catch-up] Έξαρση στο {cid}: {recent} χαρ. την τελευταία ώρα "
          f"(τυπική ώρα ~{baseline:.0f}) -> σύνοψη")

    in_progress.add(cid)
    task = asyncio.create_task(run_catchup(channel))
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)


@tasks.loop(time=AUTO_TLDR_TIMES)
async def auto_tldr_task():
    if not AUTO_TLDR_CHANNEL_ID:
        return
    try:
        channel_id = int(AUTO_TLDR_CHANNEL_ID)
        channel = bot.get_channel(channel_id) or await bot.fetch_channel(channel_id)

        if isinstance(channel, discord.TextChannel):
            header = f"🤖 **Αυτόματο TL;DR Τελευταίων {AUTO_TLDR_HOURS} Ωρών** 📝\n\n"
            payload = await make_tldr_payload(channel, AUTO_TLDR_HOURS, header)
            if payload:
                text, file = payload
                reset_catchup(channel_id)
                try:
                    if file:
                        await channel.send(text, file=file)
                    else:
                        await channel.send(text)
                except Exception as e:
                    print(f"[Auto TLDR send with image failed] {e}")
                    if file:
                        await channel.send(text)  # ξανά χωρίς εικόνα
    except Exception as e:
        print(f"[Auto TLDR Error] {e}")


@auto_tldr_task.before_loop
async def before_auto_tldr():
    await bot.wait_until_ready()


async def safe_followup(interaction: discord.Interaction, text: str, file: discord.File | None = None):
    try:
        if file:
            await interaction.followup.send(text[:1900], file=file)
        else:
            await interaction.followup.send(text[:1900])
    except Exception as e:
        print(f"[Followup Error] {e}")
        if file:  # δεύτερη προσπάθεια χωρίς εικόνα
            try:
                await interaction.followup.send(text[:1900])
            except Exception as e2:
                print(f"[Followup retry Error] {e2}")


@bot.tree.command(
    name="tldr",
    description="Δημιουργεί σύνοψη (TL;DR) των μηνυμάτων του καναλιού.",
)
@app_commands.describe(hours="Πόσες ώρες πίσω να ανατρέξει το bot (1 έως 8)")
async def tldr(interaction: discord.Interaction, hours: int):
    channel_id = interaction.channel_id
    now = time.time()

    if hours <= 0 or hours > 8:
        await interaction.response.send_message(
            "⚠️ Παρακαλώ δώσε έναν αριθμό ωρών μεταξύ **1 και 8**.", ephemeral=True
        )
        return

    if channel_id in in_progress:
        await interaction.response.send_message(
            "⏳ Ετοιμάζω ήδη σύνοψη για αυτό το κανάλι, περίμενε λίγο.", ephemeral=True
        )
        return

    if channel_id in last_used:
        elapsed = now - last_used[channel_id]
        if elapsed < COOLDOWN_SECONDS:
            available_at_unix = int(last_used[channel_id] + COOLDOWN_SECONDS)
            await interaction.response.send_message(
                f"⏳ Το `/tldr` είναι σε cooldown για αυτό το κανάλι!\n"
                f"Θα είναι ξανά διαθέσιμο <t:{available_at_unix}:R> (στις <t:{available_at_unix}:t>)."
            )
            return

    await interaction.response.defer(thinking=True)
    in_progress.add(channel_id)

    try:
        channel = interaction.channel
        if not isinstance(channel, discord.TextChannel):
            await safe_followup(interaction, "Αυτή η εντολή υποστηρίζεται μόνο σε κείμενα καναλιών.")
            return

        header = f"**TL;DR Τελευταίων {hours} Ωρών** 📝\n\n"
        payload = await make_tldr_payload(channel, hours, header)
        if not payload:
            await safe_followup(interaction, f"Δεν βρέθηκαν νέα μηνύματα τις τελευταίες {hours} ώρες.")
            return

        last_used[channel_id] = time.time()
        reset_catchup(channel_id)
        text, file = payload
        await safe_followup(interaction, text, file)

    except SummaryError as e:
        await safe_followup(interaction, f"⚠️ {e}")
    except Exception as e:
        print(f"[TLDR Error] {e!r}")
        await safe_followup(interaction, "⚠️ Κάτι πήγε στραβά. Δοκίμασε ξανά σε λίγο.")
    finally:
        in_progress.discard(channel_id)


@bot.tree.command(
    name="cooldown",
    description="Ελέγχει πόσος χρόνος απομένει για την επόμενη χρήση του /tldr.",
)
async def cooldown_status(interaction: discord.Interaction):
    channel_id = interaction.channel_id
    now = time.time()

    if channel_id in last_used:
        elapsed = now - last_used[channel_id]
        if elapsed < COOLDOWN_SECONDS:
            available_at_unix = int(last_used[channel_id] + COOLDOWN_SECONDS)
            await interaction.response.send_message(
                f"⏱️ Το `/tldr` θα είναι ξανά διαθέσιμο στο κανάλι <t:{available_at_unix}:R>.",
                ephemeral=True,
            )
            return

    await interaction.response.send_message(
        "✅ Το `/tldr` είναι **έτοιμο για χρήση** στο κανάλι!", ephemeral=True
    )


@bot.tree.command(
    name="resetcd",
    description="Μηδενίζει το cooldown της εντολής /tldr (Μόνο για Διαχειριστές).",
)
@app_commands.describe(all_channels="Καθαρισμός cooldown για όλα τα κανάλια;")
@app_commands.default_permissions(administrator=True)
async def reset_cooldown(interaction: discord.Interaction, all_channels: bool = False):
    channel_id = interaction.channel_id

    if all_channels:
        last_used.clear()
        await interaction.response.send_message(
            "🔄 Το cooldown του `/tldr` μηδενίστηκε **για όλα τα κανάλια**!", ephemeral=True
        )
    elif channel_id in last_used:
        del last_used[channel_id]
        await interaction.response.send_message(
            "🔄 Το cooldown του `/tldr` μηδενίστηκε για **αυτό το κανάλι**!", ephemeral=True
        )
    else:
        await interaction.response.send_message(
            "ℹ️ Δεν υπήρχε ενεργό cooldown σε αυτό το κανάλι.", ephemeral=True
        )


bot.run(DISCORD_TOKEN)