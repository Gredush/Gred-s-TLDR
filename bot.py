import os
from datetime import datetime, timedelta, timezone
import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv
from google import genai

# Φόρτωση τοπικού .env αν υπάρχει
load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Έλεγχος αν βρέθηκαν τα keys
if not DISCORD_TOKEN:
    print(
        "CRITICAL ERROR: Το DISCORD_TOKEN δεν βρέθηκε στις μεταβλητές περιβάλλοντος!"
    )
if not GEMINI_API_KEY:
    print(
        "CRITICAL ERROR: Το GEMINI_API_KEY δεν βρέθηκε στις μεταβλητές περιβάλλοντος!"
    )

# Αρχικοποίηση client περνώντας ρητά το key
ai_client = genai.Client(api_key=GEMINI_API_KEY)

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user.name} (ID: {bot.user.id})")
    try:
        synced = await bot.tree.sync()
        print(f"Synced {len(synced)} command(s)")
    except Exception as e:
        print(f"Sync error: {e}")


@bot.tree.command(
    name="tldr",
    description="Δημιουργεί σύνοψη (TL;DR) των μηνυμάτων του καναλιού.",
)
@app_commands.describe(hours="Πόσες ώρες πίσω να κοιτάξει (π.χ. 4)")
async def tldr(interaction: discord.Interaction, hours: int):
    await interaction.response.defer(thinking=True)

    if hours <= 0 or hours > 72:
        await interaction.followup.send("Δώσε αριθμό ωρών μεταξύ 1 και 72.")
        return

    cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
    messages_list = []

    async for message in interaction.channel.history(
        after=cutoff_time, limit=500
    ):
        if message.author.bot or not message.content.strip():
            continue
        timestamp = message.created_at.strftime("%H:%M")
        messages_list.append(
            f"[{timestamp}] {message.author.display_name}: {message.content}"
        )

    if not messages_list:
        await interaction.followup.send(
            f"Δεν βρέθηκαν νέα μηνύματα τις τελευταίες {hours} ώρες."
        )
        return

    chat_log = "\n".join(messages_list)

    prompt = f"""
Είσαι ένας βοηθός Discord bot. Παρακάτω είναι μια συνομιλία από ένα Discord κανάλι (Ελληνικά, Greeklish, Αγγλικά).
Στόχος σου είναι να φτιάξεις ένα καθαρό, δομημένο TL;DR (σύνοψη) στα Ελληνικά με bullet points.

Ιστορικό Συνομιλίας:
{chat_log}
"""

    try:
        response = ai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        summary = response.text

        header = f"**TL;DR Τελευταίων {hours} Ωρών** 📝\n\n"
        if len(header + summary) > 2000:
            summary = (
                summary[: 1900 - len(header)]
                + "...\n*(Η σύνοψη κόπηκε λόγω ορίου)*"
            )

        await interaction.followup.send(header + summary)
    except Exception as e:
        print(f"Gemini API Error: {e}")
        await interaction.followup.send(
            "Σφάλμα κατά τη δημιουργία της σύνοψης."
        )


bot.run(DISCORD_TOKEN)
