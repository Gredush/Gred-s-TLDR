import asyncio
import os
import time
from datetime import datetime, time as dt_time, timedelta, timezone
import discord
from discord import app_commands
from discord.ext import commands, tasks
from dotenv import load_dotenv
import aiohttp
from google import genai
from groq import AsyncGroq

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
AUTO_TLDR_CHANNEL_ID = os.getenv("AUTO_TLDR_CHANNEL_ID")

groq_client = AsyncGroq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

GUILD_ID = None
COOLDOWN_SECONDS = 1800
last_used = {}

SCHEDULED_TIMES = [
    dt_time(hour=10, minute=0, tzinfo=timezone.utc),
    dt_time(hour=18, minute=0, tzinfo=timezone.utc),
]

def fit_to_discord_limit(header: str, text: str, max_limit: int = 1980) -> str:
    """Διασφαλίζει ότι το συνολικό μήνυμα δεν ξεπερνά ποτέ το όριο του Discord (2000 chars)."""
    full_text = header + text
    if len(full_text) <= max_limit:
        return full_text
    cutoff_note = "\n\n*(Η σύνοψη κόπηκε λόγω ορίου χαρακτήρων)*"
    allowed_text_len = max_limit - len(header) - len(cutoff_note)
    trimmed_text = text[:allowed_text_len]
    return header + trimmed_text + cutoff_note

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user.name} (ID: {bot.user.id})")
    try:
        if GUILD_ID:
            bot.tree.copy_global_to(guild=GUILD_ID)
            synced = await bot.tree.sync(guild=GUILD_ID)
        else:
            synced = await bot.tree.sync()
        print(f"Synced {len(synced)} command(s)")
    except Exception as e:
        print(f"Sync error: {e}")
    if not auto_tldr_task.is_running():
        auto_tldr_task.start()

async def call_gemini_rest_fallback(prompt: str, model_name: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, timeout=45) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
            else:
                err_text = await resp.text()
                raise RuntimeError(f"HTTP {resp.status} ({model_name}): {err_text[:300]}")

async def generate_summary_with_fallback(system_prompt: str, chat_log: str) -> str:
    errors = []
    full_prompt = f"{system_prompt}\n\nΙστορικό Συνομιλίας:\n{chat_log}"

    # === 1. GROQ - Μόνο αξιόπιστα chat μοντέλα ===
    if groq_client:
        # Σκληρά hardcoded τα καλύτερα διαθέσιμα μοντέλα (αποφεύγουμε το dynamic list)
        groq_models = [
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "gemma2-9b-it",
            "mixtral-8x7b-32768",
        ]
        for model_name in groq_models:
            try:
                response = await groq_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Ιστορικό Συνομιλίας:\n{chat_log}"},
                    ],
                    temperature=0.4,
                    max_tokens=900,
                )
                summary = response.choices[0].message.content
                if summary and summary.strip():
                    print(f"[SUCCESS] Groq model used: {model_name}")
                    return summary
            except Exception as e:
                error_msg = str(e)[:200]
                errors.append(f"Groq ({model_name}): {error_msg}")
                print(f"[Groq Fail] {model_name}: {error_msg}")

    # === 2. GEMINI SDK ===
    if gemini_client:
        gemini_models = ["gemini-3.8-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
        for model_name in gemini_models:
            try:
                response = await gemini_client.aio.models.generate_content(
                    model=model_name,
                    contents=full_prompt,
                )
                summary = response.text
                if summary and summary.strip():
                    print(f"[SUCCESS] Gemini SDK model used: {model_name}")
                    return summary
            except Exception as e:
                error_msg = str(e)[:200]
                errors.append(f"Gemini SDK ({model_name}): {error_msg}")
                print(f"[Gemini SDK Fail] {model_name}: {error_msg}")

    # === 3. GEMINI REST FALLBACK ===
    if GEMINI_API_KEY:
        for model_name in ["gemini-3.8-flash", "gemini-2.0-flash"]:
            try:
                summary = await call_gemini_rest_fallback(full_prompt, model_name)
                if summary and summary.strip():
                    print(f"[SUCCESS] Gemini REST model used: {model_name}")
                    return summary
            except Exception as e:
                error_msg = str(e)[:200]
                errors.append(f"Gemini REST ({model_name}): {error_msg}")
                print(f"[Gemini REST Fail] {model_name}: {error_msg}")

    # Αν φτάσουμε εδώ, όλα απέτυχαν
    short_errors = "\n• ".join(errors[:6])  # κρατάμε μόνο τα πρώτα 6 errors
    raise RuntimeError(f"Αποτυχία όλων των APIs:\n• {short_errors}")

async def fetch_and_generate_tldr(channel: discord.TextChannel, hours: int) -> str | None:
    now = datetime.now(timezone.utc)
    target_cutoff = now - timedelta(hours=hours)
    context_hours = max(18, hours * 2)
    context_cutoff = now - timedelta(hours=context_hours)

    target_messages = []
    context_messages = []
    total_fetched = 0

    async for message in channel.history(limit=600):
        total_fetched += 1

        if message.created_at < context_cutoff:
            break

        if message.author.bot or not message.content.strip():
            continue

        # Κόβουμε πολύ μεγάλα μηνύματα
        content = message.content.strip()
        if len(content) > 400:
            content = content[:400] + "..."

        timestamp = message.created_at.strftime("%H:%M")
        formatted_msg = f"[{timestamp}] {message.author.display_name}: {content}"

        if message.created_at >= target_cutoff:
            target_messages.append(formatted_msg)
        else:
            context_messages.append(formatted_msg)

    print(f"[TLDR Debug] Channel: {channel.name} | Fetched: {total_fetched} | Target: {len(target_messages)} | Context: {len(context_messages)}")

    if not target_messages:
        return None

    # Αντιστρέφουμε (παλιό → νέο)
    target_messages.reverse()
    context_messages.reverse()

    # Πολύ πιο επιθετικό trimming για να μην σκάνε τα APIs
    context_log = "\n".join(context_messages)
    if len(context_log) > 3500:
        context_log = context_log[-3500:]

    target_log = "\n".join(target_messages)
    if len(target_log) > 5500:
        target_log = target_log[-5500:]

    system_prompt = (
        "Είσαι ένας γραμματέας Discord. Διαβάζεις συνομιλίες (Ελληνικά, Greeklish, Αγγλικά) "
        "και φτιάχνεις δομημένη, καθαρή σύνοψη (TL;DR) στα Ελληνικά.\n\n"
        "Οδηγίες:\n"
        "1. Χώρισε σε θεματικές ενότητες με έντονα γράμματα.\n"
        "2. Χρησιμοποίησε bullet points όπου βοηθάει.\n"
        "3. Σημείωσε σημαντικές αποφάσεις στο τέλος.\n"
        "4. ΜΗΝ βάζεις links από εικόνες/gifs/media.\n"
        "5. ΚΡΑΤΑ ΤΗ ΣΥΝΟΨΗ ΣΥΝΤΟΜΗ (κάτω από 220 λέξεις).\n"
        "6. Μόνο τι ειπώθηκε – χωρίς δικά σου συμπεράσματα.\n"
        "7. Ουδέτερος και αντικειμενικός.\n"
        "8. Απαντάς ΜΟΝΟ στα Ελληνικά."
    )

    full_chat_payload = (
        f"--- ΠΡΟΗΓΟΥΜΕΝΟ CONTEXT ---\n"
        f"{context_log if context_log else 'Δεν υπάρχει προηγούμενο context.'}\n\n"
        f"--- ΜΗΝΥΜΑΤΑ ΠΡΟΣ ΣΥΝΟΨΗ (τελευταίες {hours} ώρες) ---\n"
        f"{target_log}"
    )

    return await generate_summary_with_fallback(system_prompt, full_chat_payload)

@tasks.loop(time=SCHEDULED_TIMES)
async def auto_tldr_task():
    if not AUTO_TLDR_CHANNEL_ID:
        return
    try:
        channel_id = int(AUTO_TLDR_CHANNEL_ID)
        channel = bot.get_channel(channel_id)
        if not channel:
            channel = await bot.fetch_channel(channel_id)
        if isinstance(channel, discord.TextChannel):
            summary = await fetch_and_generate_tldr(channel, hours=8)
            if summary:
                header = "🤖 **Αυτόματο TL;DR Τελευταίων 8 Ωρών** 📝\n\n"
                final_msg = fit_to_discord_limit(header, summary)
                await channel.send(final_msg)
    except Exception as e:
        print(f"[Auto TLDR Error] {e}")

@auto_tldr_task.before_loop
async def before_auto_tldr():
    await bot.wait_until_ready()

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
            "⚠️ Παρακαλώ δώσε έναν αριθμό ωρών μεταξύ **1 και 8**.",
            ephemeral=True,
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

    try:
        channel = interaction.channel
        if isinstance(channel, discord.TextChannel):
            summary = await fetch_and_generate_tldr(channel, hours=hours)
            if not summary:
                await interaction.followup.send(
                    f"Δεν βρέθηκαν νέα μηνύματα τις τελευταίες {hours} ώρες."
                )
                return

            last_used[channel_id] = time.time()
            header = f"**TL;DR Τελευταίων {hours} Ωρών** 📝\n\n"
            final_msg = fit_to_discord_limit(header, summary)
            await interaction.followup.send(final_msg)
        else:
            await interaction.followup.send("Αυτή η εντολή υποστηρίζεται μόνο σε κείμενα καναλιών.")
    except Exception as e:
        print(f"[TLDR Error] {e}")
        # Στέλνουμε σύντομο μήνυμα για να μην σκάσει το Discord
        error_text = str(e)
        if len(error_text) > 1500:
            error_text = error_text[:1500] + "..."
        await interaction.followup.send(
            f"Υπήρξε πρόβλημα με τις υπηρεσίες AI:\n```{error_text}```"
        )

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
        "✅ Το `/tldr` είναι **έτοιμο για χρήση** στο κανάλι!",
        ephemeral=True,
    )

bot.run(DISCORD_TOKEN)
