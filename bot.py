import os
from datetime import datetime, timedelta, timezone
import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv
from google import genai

# Φόρτωση περιβαλλοντικών μεταβλητών
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Αρχικοποίηση Gemini Client
ai_client = genai.Client(api_key=GEMINI_API_KEY)

# Intents για το Discord Bot
intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user.name} (ID: {bot.user.id})")
    try:
        synced = await bot.tree.sync()
        print(f"Synced {len(synced)} slash command(s)")
    except Exception as e:
        print(f"Error syncing commands: {e}")


@bot.tree.command(
    name="tldr",
    description="Δημιουργεί σύνοψη (TL;DR) των μηνυμάτων του καναλιού για το χρονικό διάστημα που ορίζεις.",
)
@app_commands.describe(
    hours="Πόσες ώρες πίσω θέλεις να ανατρέξει το bot (π.χ. 2, 8, 24)"
)
async def tldr(interaction: discord.Interaction, hours: int):
    # Ενημέρωση του Discord ότι η επεξεργασία θα πάρει λίγο χρόνο
    await interaction.response.defer(thinking=True)

    if hours <= 0 or hours > 72:
        await interaction.followup.send(
            "Παρακαλώ δώσε αριθμό ωρών μεταξύ 1 και 72."
        )
        return

    # Υπολογισμός χρονικού ορίου
    cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

    messages_list = []
    async for message in interaction.channel.history(
        after=cutoff_time, limit=500
    ):
        # Αγνοούμε μηνύματα από bots
        if message.author.bot:
            continue
        # Παράκαμψη κενών μηνυμάτων (π.χ. μόνο εικόνες/embeds)
        if not message.content.strip():
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

    # Prompt για το LLM
    prompt = f"""
Είσαι ένας βοηθός Discord bot. Παρακάτω είναι μια συνομιλία από ένα Discord κανάλι που περιέχει ελληνικά, greeklish και αγγλικά.

Στόχος σου είναι να φτιάξεις ένα καθαρό, δομημένο TL;DR (σύνοψη) στα Ελληνικά για να καταλάβει ο χρήστης τι έχασε όσο έλειπε.

Οδηγίες:
- Ομάδοποίησε τα κύρια θέματα συζήτησης σε bullet points.
- Αναέφερε ποιοι χρήστες συμμετείχαν στα βασικά θέματα αν είναι σχετικό.
- Αν υπήρχαν σημαντικές αποφάσεις, links ή ανακοινώσεις, σημείωσέ τες.
- Κράτα το ύφος φιλικό και σύντομο.

Ιστορικό Συνομιλίας:
{chat_log}
"""

    try:
        response = ai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        summary = response.text

        # Όριο Discord μηνύματος: 2000 χαρακτήρες
        header = f"**TL;DR Τελευταίων {hours} Ωρών** 📝\n\n"
        if len(header + summary) > 2000:
            summary = summary[: 1900 - len(header)] + "...\n*(Η σύνοψη κόπηκε λόγω ορίου χαρακτήρων)*"

        await interaction.followup.send(header + summary)

    except Exception as e:
        print(f"Error generating summary: {e}")
        await interaction.followup.send(
            "Υπήρξε σφάλμα κατά τη δημιουργία της σύνοψης. Παρακαλώ δοκίμασε ξανά."
        )


bot.run(DISCORD_TOKEN)