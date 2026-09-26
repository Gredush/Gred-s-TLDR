import os
import time
from datetime import datetime, timedelta, timezone, time as dtime
from zoneinfo import ZoneInfo
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

# Ώρα Ελλάδας. Το tasks.loop(time=...) υπολογίζει το επόμενο τρέξιμο με βάση
# το ρολόι (όχι πότε ξεκίνησε η διεργασία), οπότε ένα restart / redeploy
# ΔΕΝ ξαναρχίζει κανένα μετρητή -- απλά περιμένει την επόμενη προγραμματισμένη ώρα.
ATHENS_TZ = ZoneInfo("Europe/Athens")
AUTO_TLDR_HOURS = 4  # κάθε πόσες ώρες, ξεκινώντας από τις 9πμ
AUTO_TLDR_TIMES = [
    dtime(hour=(9 + AUTO_TLDR_HOURS * i) % 24, minute=0, tzinfo=ATHENS_TZ)
    for i in range(24 // AUTO_TLDR_HOURS)
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
            print(f"Synced {len(synced)} command(s) to Guild {GUILD_ID.id}")
        else:
            synced = await bot.tree.sync()
            print(f"Synced {len(synced)} command(s) globally.")
    except Exception as e:
        print(f"Sync error: {e}")

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
                    temperature=0.3,
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


async def fetch_and_generate_tldr(channel: discord.TextChannel, hours: int) -> str | None:
    cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
    messages_list = []

    # Διαβάζουμε ΑΠΟ ΤΟ ΠΙΟ ΠΡΟΣΦΑΤΟ ΠΡΟΣ ΤΑ ΠΙΣΩ (oldest_first=False).
    # Έτσι, αν υπάρχουν περισσότερα από `limit` μηνύματα στο χρονικό
    # παράθυρο, αυτά που κόβονται είναι τα παλιότερα -- όχι τα πιο πρόσφατα.
    # Με after=... και την προεπιλεγμένη σειρά (oldest_first) θα γέμιζε το
    # όριο με τα παλιότερα μηνύματα και θα σταματούσε πριν φτάσει στο τώρα.
    async for message in channel.history(after=cutoff_time, limit=500, oldest_first=False):
        if message.author.bot or not message.content.strip():
            continue

        timestamp = message.created_at.strftime("%H:%M")
        messages_list.append(
            f"[{timestamp}] {message.author.display_name}: {message.content}"
        )

    if not messages_list:
        return None

    # Τα γυρίζουμε σε χρονολογική σειρά (παλιό -> νέο) πριν φτιάξουμε το log.
    messages_list.reverse()

    chat_log = "\n".join(messages_list)
    if len(chat_log) > 15000:
        chat_log = chat_log[-15000:]

    system_prompt = (
        "Είσαι ένας εξαιρετικά ακριβής, αντικειμενικός και επαγγελματίας αναλυτής συνομιλιών Discord.\n"
        "Η αποστολή σου είναι να διαβάζεις ιστορικά συνομιλιών (που περιέχουν Ελληνικά, Greeklish, Αγγλικά και slang) "
        "και να συντάσσεις μια καθαρή, άρτια δομημένη και απόλυτα αντικειμενική σύνοψη (TL;DR) αποκλειστικά στα Ελληνικά.\n\n"
        "### ΚΑΝΟΝΕΣ ΕΠΕΞΕΡΓΑΣΙΑΣ ΚΕΙΜΕΝΟΥ:\n"
        "1. **ΑΥΣΤΗΡΗ ΑΝΤΙΚΕΙΜΕΝΙΚΟΤΗΤΑ (NO HALLUCINATIONS):** Σύνοψισε ΑΠΟΚΛΕΙΣΤΙΚΑ τα γεγονότα και τις τοποθετήσεις που αναφέρονται ρητά. "
        "ΜΗΝ κάνεις υποθέσεις, μην συμπληρώνεις 'κενά' στη συζήτηση με δικά σου σενάρια και μην βγάζεις αυθαίρετα συμπεράσματα.\n"
        "2. **ΓΛΩΣΣΑ & ΓΡΑΜΜΑΤΙΚΗ:** Γράψε σε άπταιστα Ελληνικά με σωστή συντακτική δομή και ορθογραφία. "
        "3. **ΟΥΔΕΤΕΡΟ ΥΦΟΣ:** Διατήρησε επαγγελματικό, δημοσιογραφικό και ουδέτερο τόνο. Μην κρίνεις τη συμπεριφορά ή τις απόψεις των χρηστών.\n"
        "4. **ΚΑΘΑΡΙΣΜΟΣ MEDIA:** Αγνόησε συνδέσμους (URLs), εικόνες, GIFs, αλληλουχίες από emojis ή αυτόματα μηνύματα bots.\n\n"
        "5. **ΔΙΑΤΗΡΗΣΗ ΣΥΝΑΦΕΙΑΣ (CONTEXT):** Μην παραθέτεις απλά απομονωμένες ατάκες. Εξήγησε την εξέλιξη της συζήτησης. "
        "Δώσε στο αναγνώστη να καταλάβει ΠΩΣ ξεκίνησε ένα θέμα, ΤΙ ειπώθηκε ενδιάμεσα και ΠΟΥ κατέληξε.\n"
        "6. **ΠΛΗΡΟΤΗΤΑ & ΛΕΠΤΟΜΕΡΕΙΑ:** Μην κάνεις τη σύνοψη τηλεγραφική ή υπερβολικά σύντομη. "
        "7. **ΦΙΛΤΡΑΡΙΣΜΑ:** Αγνόησε τελείως links που παραπέμπουν σε gifs, εικόνες, emojis χωρίς νόημα αλλά συγκράτησε links τα οποία συνεισφέρουν στην συζήτηση.\n\n"
        "Συμπεριέλαβε όλες τις ουσιαστικές λεπτομέρειες, τα επιχειρήματα ή τις λεπτομέρειες που αναφέρθηκαν στο chatroom.\n"
        "### ΔΟΜΗ ΕΞΟΔΟΥ (Χρησιμοποίησε ακριβώς αυτή τη μορφοποίηση):\n\n"
        "📌 **Κύρια Θέματα Συζήτησης**\n"
        "• **[Όνομα Θέματος 1]:** Περιγραφή του τι συζητήθηκε και από ποιους (αν είναι σχετικό).Εξήγησε το πλαίσιο, τι αναφέρθηκε από τους συμμετέχοντες και πώς εξελίχθηκε η κουβέντα.\n"
        "• **[Όνομα Θέματος 2]:** Περιγραφή του τι συζητήθηκε και από ποιους με την ίδια λογική.\n\n"
        "💡 **Σημαντικές Αποφάσεις / Συμπεράσματα**\n"
        "• (Κατέγραψε τυχόν αποφάσεις που λήφθηκαν. Αν δεν λήφθηκε καμία απόφαση, γράψε 'Καμία απόφαση').\n\n"
        "📋 **Επόμενα Βήματα / Action Items**\n"
        "• (Τυχόν εργασίες, ραντεβού ή ενέργειες που συμφωνήθηκαν. Αν δεν υπάρχουν, παράλειψε αυτή την ενότητα).\n\n"
        "### ΠΕΡΙΟΡΙΣΜΟΣ ΜΗΚΟΥΣ:\n"
        "Κράτα τη συνολική απάντηση σύντομη και ευανάγνωστη (περίπου 200-250 λέξεις)."
    )

    return await generate_summary_with_fallback(system_prompt, chat_log)


@tasks.loop(time=AUTO_TLDR_TIMES)
async def auto_tldr_task():
    """Τρέχει σε συγκεκριμένες ώρες ρολογιού (π.χ. 9:00, 13:00, 17:00, 21:00,
    1:00, 5:00 ώρα Ελλάδας), όχι σε rolling interval από την εκκίνηση. Επειδή
    το discord.py υπολογίζει την επόμενη εκτέλεση με βάση το ρολόι, ένα
    restart/redeploy απλά περιμένει κανονικά την επόμενη προγραμματισμένη ώρα
    -- δεν ξεκινάει νέο 4ωρο μέτρημα από την αρχή."""
    if not AUTO_TLDR_CHANNEL_ID:
        return

    try:
        channel_id = int(AUTO_TLDR_CHANNEL_ID)
        channel = bot.get_channel(channel_id)
        if not channel:
            channel = await bot.fetch_channel(channel_id)

        if isinstance(channel, discord.TextChannel):
            summary = await fetch_and_generate_tldr(channel, hours=AUTO_TLDR_HOURS)
            if summary:
                header = f"🤖 **Αυτόματο TL;DR Τελευταίων {AUTO_TLDR_HOURS} Ωρών** 📝\n\n"
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
