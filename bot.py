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
SUMMARY_CHANNEL_ID = os.getenv("SUMMARY_CHANNEL_ID")  # ID Καναλιού για το πρωινό TL;DR

# Αρχικοποίηση SDK Clients
groq_client = AsyncGroq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)
GUILD_ID = None

# Διάρκεια Cooldown σε δευτερόλεπτα (30 λεπτά)
COOLDOWN_SECONDS = 1800

# Dictionary για παρακολούθηση του τελευταίου timestamp ανά κανάλι: {channel_id: last_used_timestamp}
last_used = {}


# ---------------------------------------------------------
# AUTO DAILY SUMMARY TASK (10:00 AM UTC+3 / Greece Time)
# ---------------------------------------------------------
@tasks.loop(minutes=1)
async def daily_summary_task():
    """Ελέγχει κάθε λεπτό αν είναι 10:00 π.μ. (UTC+3) για να στείλει το ημερήσιο TL;DR."""
    if not SUMMARY_CHANNEL_ID:
        return

    # Ώρα Ελλάδος (UTC+3)
    tz_greece = timezone(timedelta(hours=3))
    now = datetime.now(tz_greece)

    # Trigger ακριβώς στις 10:00:00 (στο 0ο λεπτό)
    if now.hour == 10 and now.minute == 0:
        channel = bot.get_channel(int(SUMMARY_CHANNEL_ID))
        if not channel:
            print(f"[Daily Summary Error] Δεν βρέθηκε το κανάλι ID: {SUMMARY_CHANNEL_ID}")
            return

        print(f"[Daily Summary] Εκτέλεση ημερήσιας σύνοψης για το κανάλι #{channel.name}...")
        try:
            summary_embed = await process_tldr_logic(channel, hours=24)

            # Extra header embed για την πρωινή ανακοίνωση
            announcement_embed = discord.Embed(
                title="☀️ Καλημέρα! Η Ημερήσια Σύνοψη είναι έτοιμη",
                description="Ορίστε τι συνέβη στο κανάλι τις τελευταίες **24 ώρες**:",
                color=discord.Color.gold()
            )
            await channel.send(embeds=[announcement_embed, summary_embed])
            print("[Daily Summary Success] Η ημερήσια σύνοψη απεστάλη επιτυχώς!")
        except Exception as e:
            print(f"[Daily Summary Exception] {e}")


@daily_summary_task.before_loop
async def before_daily_summary():
    await bot.wait_until_ready()


# ---------------------------------------------------------
# BOT EVENTS & HELPER FUNCTIONS
# ---------------------------------------------------------
@bot.event
async def on_ready():
    print(f"Logged in as {bot.user.name} (ID: {bot.user.id})")

    # Εκκίνηση του daily task
    if not daily_summary_task.is_running():
        daily_summary_task.start()
        print("[Task Manager] Το Daily Summary Loop ξεκίνησε!")

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


async def get_active_groq_models() -> list[str]:
    """Φέρνει δυναμικά τα ενεργά μοντέλα από την Groq."""
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
    """Φέρνει δυναμικά τα ενεργά μοντέλα από το Gemini SDK."""
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
    """Direct REST HTTP Call στο Gemini API ως έσχατη λύση."""
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
                    max_tokens=1500,
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


async def process_tldr_logic(channel: discord.TextChannel, hours: int, status_message: discord.WebhookMessage = None) -> discord.Embed:
    """Κύρια λογική συλλογής μηνυμάτων & παραγωγής Embed σύνοψης."""
    if status_message:
        await status_message.edit(content="🔍 **Στάδιο 1/2:** Συλλογή μηνυμάτων καναλιού...")

    cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
    messages_list = []

    async for message in channel.history(after=cutoff_time, limit=500):
        if message.author.bot or not message.content.strip():
            continue

        timestamp = message.created_at.strftime("%H:%M")
        messages_list.append(f"[{timestamp}] {message.author.display_name}: {message.content}")

    if not messages_list:
        embed = discord.Embed(
            title="📭 Δεν βρέθηκαν μηνύματα",
            description=f"Δεν υπήρξε δραστηριότητα στο κανάλι τις τελευταίες **{hours}** ώρες.",
            color=discord.Color.orange()
        )
        return embed

    if status_message:
        await status_message.edit(content=f"🧠 **Στάδιο 2/2:** Ανάλυση {len(messages_list)} μηνυμάτων με AI...")

    chat_log = "\n".join(messages_list)
    if len(chat_log) > 15000:
        chat_log = chat_log[-15000:]

    system_prompt = (
        "Είσαι ένας βοηθός Discord bot. Η δουλειά σου είναι να διαβάζεις"
        " συνομιλίες (που περιέχουν Ελληνικά, Greeklish και Αγγλικά) και να"
        " φτιάχνεις μια δομημένη και ελαφρώς αναλυτική σύνοψη (TL;DR) στα Ελληνικά.\n\n"
        "Οδηγίες Δομής:\n"
        "1. Χώρισε τη σύνοψη σε ξεχωριστές θεματικές ενότητες (με έντονα γράμματα) για κάθε κύριο θέμα συζήτησης.\n"
        "2. Σε κάθε θέμα, εξήγησε με 2-3 παραστατικά bullet points τι ακριβώς ειπώθηκε, ποιες ήταν οι απόψεις ή τα επιχειρήματα των χρηστών (αναφέροντας τα ονόματά τους) και αν υπήρξε κάποιο συμπέρασμα.\n"
        "3. Αν υπήρξαν σημαντικές αποφάσεις ή εκκρεμότητες, σημείωσέ τες στο τέλος.\n"
        "4. ΜΗΝ περιλαμβάνεις σύνδεσμους (links) από εικόνες, gifs ή πολυμέσα που κοινοποιήθηκαν στο chat.\n"
        "5. Κράτα το ύφος φυσικό, φιλικό και ευανάγνωστο, αποφεύγοντας υπερβολικά σύντομες προτάσεις."
    )

    summary = await generate_summary_with_fallback(system_prompt, chat_log)

    # Δημιουργία Discord Embed
    embed = discord.Embed(
        title=f"📝 TL;DR - Τελευταίες {hours} Ώρες",
        description=summary if len(summary) <= 4000 else summary[:3900] + "\n\n*(Η σύνοψη κόπηκε λόγω ορίου)*",
        color=discord.Color.blurple(),
        timestamp=datetime.now(timezone.utc)
    )
    embed.set_footer(text=f"Αναλύθηκαν {len(messages_list)} μηνύματα • Powered by AI")
    return embed


# ---------------------------------------------------------
# SLASH COMMANDS
# ---------------------------------------------------------
@bot.tree.command(
    name="tldr",
    description="Δημιουργεί σύνοψη (TL;DR) των μηνυμάτων του καναλιού.",
)
@app_commands.describe(hours="Πόσες ώρες πίσω να ανατρέξει το bot (1 έως 72)")
async def tldr(interaction: discord.Interaction, hours: int):
    channel_id = interaction.channel_id
    now = time.time()

    # Έλεγχος Cooldown
    if channel_id in last_used:
        elapsed = now - last_used[channel_id]
        if elapsed < COOLDOWN_SECONDS:
            available_at_unix = int(last_used[channel_id] + COOLDOWN_SECONDS)
            await interaction.response.send_message(
                f"⏳ Το `/tldr` είναι σε cooldown για αυτό το κανάλι!\n"
                f"Θα είναι ξανά διαθέσιμο <t:{available_at_unix}:R> (στις <t:{available_at_unix}:t>)."
            )
            return

    if hours <= 0 or hours > 72:
        await interaction.response.send_message(
            "Παρακαλώ δώσε έναν αριθμό ωρών μεταξύ 1 και 72.", ephemeral=True
        )
        return

    # Defer με live message update
    await interaction.response.defer(thinking=True)
    status_msg = await interaction.original_response()

    try:
        summary_embed = await process_tldr_logic(interaction.channel, hours, status_message=status_msg)

        # Ενημέρωση timestamp μόνο αν παράχθηκε επιτυχώς σύνοψη
        last_used[channel_id] = time.time()

        # Καθαρισμός του status text και αποστολή του Embed
        await interaction.followup.send(embed=summary_embed)
        await status_msg.delete()  # Διαγράφει το προσωρινό μήνυμα κατάστασης

    except Exception as e:
        print(f"TLDR Error: {e}")
        await interaction.followup.send(
            f"❌ Υπήρξε πρόβλημα με την επεξεργασία: `{e}`"
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
