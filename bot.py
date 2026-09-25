import os
from datetime import datetime, timedelta, timezone
import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv
from google import genai

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not DISCORD_TOKEN or not GEMINI_API_KEY:
    print("CRITICAL ERROR: Λείπουν τα API Keys!")

ai_client = genai.Client(api_key=GEMINI_API_KEY)

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

GUILD_ID = discord.Object(
    id=123456789012345678
)  # Βάλε το Server ID σου αν θες άμεσο sync


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user.name} (ID: {bot.user.id})")
    try:
        bot.tree.copy_global_to(guild=GUILD_ID)
        synced = await bot.tree.sync(guild=GUILD_ID)
        print(
            f"Synced {len(synced)} command(s) directly to guild {GUILD_ID.id}"
        )
    except Exception as e:
        print(f"Sync error: {e}")


@bot.tree.command(
    name="tldr",
    description="Δημιουργεί σύνοψη (TL;DR) των μηνυμάτων του καναλιού.",
)
@app_commands.describe(hours="Πόσες ώρες πίσω να κοιτάξει (1 έως 72)")
async def tldr(interaction: discord.Interaction, hours: int):
    await interaction.response.defer(thinking=True)

    if hours <= 0 or hours > 72:
        await interaction.followup.send("Δώσε αριθμό ωρών μεταξύ 1 και 72.")
        return

    cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
    messages_list = []

    async for message in interaction.channel.history(
        after=cutoff_time, limit=300
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

    # Περιορισμός μεγέθους κειμένου αν υπάρχουν πάρα πολλά μηνύματα
    chat_log = "\n".join(messages_list)
    if len(chat_log) > 15000:
        chat_log = chat_log[-15000:]  # Κρατάμε τα πιο πρόσφατα

    prompt = f"""
Είσαι ένας βοηθός Discord bot. Παρακάτω είναι μια συνομιλία από ένα Discord κανάλι που περιέχει Ελληνικά, Greeklish και Αγγλικά.
Στόχος σου είναι να φτιάξεις ένα καθαρό, δομημένο TL;DR (σύνοψη) στα Ελληνικά με bullet points.

Ιστορικό Συνομιλίας:
{chat_log}
"""

    try:
        # Χρήση του σταθερού gemini-1.5-flash
        response = ai_client.models.generate_content(
            model="gemini-1.5-flash",
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
        # Τυπώνουμε το ακριβές σφάλμα στα Deploy Logs του Railway
        print(f"DETAILED GEMINI ERROR: {type(e).__name__}: {e}")
        await interaction.followup.send(
            f"Υπήρξε σφάλμα κατά τη επικοινωνία με το AI API: `{e}`"
        )


bot.run(DISCORD_TOKEN)
