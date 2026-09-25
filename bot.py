import os
import time
from datetime import datetime, timedelta, timezone
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

# Ορισμός τοπικής ζώνης ώρας (Ελλάδα / EEST / EET)
# Ή αν ο server τρέχει σε UTC, μπορείς να αλλάξεις τις ώρες αντίστοιχα.
# Εδώ ορίζουμε 10:00 και 18:00 σε UTC (προσαρμόζεις αν ο server έχει τοπική ώρα).
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
    except Exception as e:
        print(f"Sync error: {e}")

    # Ξεκινάει το task, αλλά περιμένει 8 ώρες πριν την πρώτη εκτέλεση!
    if not auto_tldr_task.is_running():
        auto_tldr_task.start()


async def get_active_groq_models() -> list[str]:
    if not groq_client:
        return []
    try:
        models_page = await groq_client.models.list()
        active_models = [m.id for m in models_page.data if m.active]
        if active_models:
            return active_models
    except Exception as e:
        print(f"[Groq ListModels Error] {e}")
    return ["llama-3.5-70b-versatile", "llama-3.3-70b-specdec", "llama3-70b-8192"]


async def get_active_gemini_models() -> list[str]:
    if not gemini_client:
        return []
    try:
        models_list = []
        async for m in gemini_client.aio.models.list():
            model_id = m.name.replace("models/", "")
            if "flash" in model_id or "generateContent" in getattr(m, "supported_generation_methods", []):
                models_list.append(model_id)
        if models_list:
            return models_list
    except Exception as e:
        print(f"[Gemini ListModels Error] {e}")
    return ["gemini-3.8-flash", "gemini-2.5-flash"]


async def call_gemini_rest_fallback(prompt: str, model_name: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, timeout=30) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
            else:
                err_text = await resp.text()
                raise RuntimeError(f"HTTP {resp.status} ({model_name}): {err_text}")


async def generate_summary_with_fallback(system_prompt: str, chat_log: str) -> str:
    errors = []
    full_prompt = f"{system_prompt}\n\nΙστορικό Συνομιλίας:\n{chat_log}"

    # 1. GROQ SDK
    if groq_client:
        groq_models = await get_active_groq_models()
        for model_name in groq_models:
            try:
                response = await groq_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Ιστορικό Συνομιλίας:\n{chat_log}"},
                    ],
                    temperature=0.5,
                    max_tokens=1000,
                )
                summary = response.choices[0].message.content
                if summary and summary.strip():
                    return summary
            except Exception as e:
                errors.append(f"Groq ({model_name}): {e}")

    # 2. GEMINI SDK
    if gemini_client:
        gemini_models = await get_active_gemini_models()
        for model_name in gemini_models:
            try:
                response = await gemini_client.aio.models.generate_content(
                    model=model_name,
                    contents=full_prompt,
                )
                summary = response.text
                if summary and summary.strip():
                    return summary
            except Exception as e:
                errors.append(f"Gemini SDK ({model_name}): {e}")

    # 3. GEMINI REST FALLBACK
    if GEMINI_API_KEY:
        for model_name in ["gemini-3.8-flash", "gemini-2.5-flash"]:
            try:
                summary = await call_gemini_rest_fallback(full_prompt, model_name)
                if summary and summary.strip():
                    return summary
            except Exception as e:
                errors.append(f"Gemini REST ({model_name}): {e}")

    all_errors_str = "\n• ".join(errors)
    raise RuntimeError(f"Αποτυχία όλων των APIs:\n• {all_errors_str}")


async def fetch_and_generate_tldr(channel: discord.TextChannel, hours: int) -> str | None:
    now = datetime.now(timezone.utc)
    
    # 1. Καθορισμός χρονικών ορίων
    target_cutoff = now - timedelta(hours=hours)
    # Το context θα είναι 24 ώρες πίσω (ή τουλάχιστον διπλάσιο αν ο χρήστης ζητήσει >12 ώρες)
    context_hours = max(24, hours * 2)
    context_cutoff = now - timedelta(hours=context_hours)

    target_messages = []
    context_messages = []

    # Μαζεύουμε μηνύματα μέχρι το context_cutoff (π.χ. 24 ώρες)
    async for message in channel.history(after=context_cutoff, limit=600):
        if message.author.bot or not message.content.strip():
            continue

        timestamp = message.created_at.strftime("%H:%M")
        formatted_msg = f"[{timestamp}] {message.author.display_name}: {message.content}"

        if message.created_at >= target_cutoff:
            target_messages.append(formatted_msg)
        else:
            context_messages.append(formatted_msg)

    # Αν δεν υπάρχουν νέα μηνύματα στο χρονικό παράθυρο που ζητήθηκε, επιστρέφουμε None
    if not target_messages:
        return None

    # Προετοιμασία κειμένων με όριο χαρακτήρων για να μην ξεπεράσουμε τα tokens
    context_log = "\n".join(context_messages)
    if len(context_log) > 8000:
        context_log = context_log[-8000:]  # Κρατάμε τα πιο πρόσφατα του context

    target_log = "\n".join(target_messages)
    if len(target_log) > 10000:
        target_log = target_log[-10000:]

    system_prompt = (
        "Είσαι ένας γραμματέας Discord. Η δουλειά σου είναι να διαβάζεις"
        " συνομιλίες (που περιέχουν Ελληνικά, Greeklish και Αγγλικά) και να"
        " φτιάχνεις μια δομημένη, καθαρή και ξεκάθαρη σύνοψη (TL;DR) στα Ελληνικά.\n\n"
        "Οδηγίες Δομής:\n"
        "1. Χώρισε τη σύνοψη σε θεματικές ενότητες (με έντονα γράμματα) για τα κύρια θέματα συζήτησης.\n"
        "2. Χώρισε το κείμενο σε σύντομες παραγράφους ή bullet points για καλύτερη αναγνωσιμότητα αν χρειάζεται.\n"
        "3. Αν υπήρξαν σημαντικές αποφάσεις, σημείωσέ τες στο τέλος.\n"
        "4. ΜΗΝ περιλαμβάνεις συνδέσμους (links) από εικόνες, gifs ή media.\n"
        "5. ΚΡΑΤΑ ΤΗ ΣΥΝΟΨΗ ΣΥΝΤΟΜΗ (κάτω από 250-280 λέξεις συνολικά).\n"
        "6. Σύνοψισε μόνο τι ειπώθηκε και τι συνέβη στο chat, χωρίς να βγάζεις δικά σου συμπεράσματα, κρίσεις ή ερμηνείες.\n"
        "7. Μείνε όσο πιο ουδέτερος και αντικειμενικός γίνεται.\n"
        "8. Απαντάς ΜΟΝΟ στα Ελληνικά.\n"
        
    )

    full_chat_payload = (
        f"--- ΠΡΟΗΓΟΥΜΕΝΟ CONTEXT (ΓΙΑ ΚΑΤΑΝΟΗΣΗ ΥΠΟΒΑΘΡΟΥ) ---\n"
        f"{context_log if context_log else 'Δεν υπάρχει προηγούμενο context.'}\n\n"
        f"--- ΝΕΑ ΜΗΝΥΜΑΤΑ ΠΡΟΣ ΣΥΝΟΨΗ (ΤΕΛΕΥΤΑΙΑ/ΕΣ {hours} ΩΡΑ/ΕΣ) ---\n"
        f"{target_log}"
    )

    return await generate_summary_with_fallback(system_prompt, full_chat_payload)
    return await generate_summary_with_fallback(system_prompt, chat_log)


@tasks.loop(hours=8)
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


import asyncio

@auto_tldr_task.before_loop
async def before_auto_tldr():
    await bot.wait_until_ready()
    # Περίμενε 8 ώρες (8 * 3600 δευτερόλεπτα) πριν την πρώτη αυτόματη εκτέλεση
    await asyncio.sleep(8 * 3600)

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
        print(f"Fallback Chain Exhausted: {e}")
        await interaction.followup.send(
            f"Υπήρξε πρόβλημα με τις υπηρεσίες AI: `{e}`"
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
