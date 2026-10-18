import asyncio
import os
import time
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

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")  # προαιρετικό
AUTO_TLDR_CHANNEL_ID = os.getenv("AUTO_TLDR_CHANNEL_ID")


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
# Πάροχοι AI
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


async def fetch_and_generate_tldr(channel: discord.TextChannel, hours: int) -> str | None:
    cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
    lines: list[str] = []

    # Από το πιο πρόσφατο προς τα πίσω, ώστε αν κοπεί κάτι να είναι τα παλιότερα.
    async for message in channel.history(after=cutoff_time, limit=750, oldest_first=False):
        content = message.clean_content.strip()
        if message.author.bot or not content:
            continue
        ts = message.created_at.astimezone(ATHENS_TZ).strftime("%H:%M")
        lines.append(f"[{ts}] {message.author.display_name}: {content[:600]}")

    if not lines:
        return None

    lines.reverse()  # παλιό -> νέο

    system_prompt = BASE_PROMPT
    if channel.id == SPICY_CHANNEL_ID:
        system_prompt += SPICY_ADDON

    return await generate_summary(system_prompt, lines)


# ------------------------------------------------------------------
# Discord
# ------------------------------------------------------------------
@bot.event
async def on_ready():
    print(f"Logged in as {bot.user.name} (ID: {bot.user.id})")
    print(f"Providers: {[p.key for p in PROVIDERS]}")
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


@tasks.loop(time=AUTO_TLDR_TIMES)
async def auto_tldr_task():
    if not AUTO_TLDR_CHANNEL_ID:
        return
    try:
        channel_id = int(AUTO_TLDR_CHANNEL_ID)
        channel = bot.get_channel(channel_id) or await bot.fetch_channel(channel_id)

        if isinstance(channel, discord.TextChannel):
            summary = await fetch_and_generate_tldr(channel, hours=AUTO_TLDR_HOURS)
            if summary:
                header = f"🤖 **Αυτόματο TL;DR Τελευταίων {AUTO_TLDR_HOURS} Ωρών** 📝\n\n"
                await channel.send(fit_to_discord_limit(header, summary))
    except Exception as e:
        print(f"[Auto TLDR Error] {e}")


@auto_tldr_task.before_loop
async def before_auto_tldr():
    await bot.wait_until_ready()


async def safe_followup(interaction: discord.Interaction, text: str):
    try:
        await interaction.followup.send(text[:1900])
    except Exception as e:
        print(f"[Followup Error] {e}")


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

        summary = await fetch_and_generate_tldr(channel, hours=hours)
        if not summary:
            await safe_followup(interaction, f"Δεν βρέθηκαν νέα μηνύματα τις τελευταίες {hours} ώρες.")
            return

        last_used[channel_id] = time.time()
        header = f"**TL;DR Τελευταίων {hours} Ωρών** 📝\n\n"
        await safe_followup(interaction, fit_to_discord_limit(header, summary))

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
