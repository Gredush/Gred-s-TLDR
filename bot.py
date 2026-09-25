import os
from datetime import datetime, timedelta, timezone
import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv
from google import genai

# Φόρτωση μεταβλητών περιβάλλοντος (για τοπική ανάπτυξη)
load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Έλεγχος αν υπάρχουν τα απαραίτητα API Keys
if not DISCORD_TOKEN:
    print("CRITICAL ERROR: Το DISCORD_TOKEN δεν βρέθηκε στις μεταβλητές περιβάλλοντος!")
if not GEMINI_API_KEY:
    print("CRITICAL ERROR: Το GEMINI_API_KEY δεν βρέθηκε στις μεταβλητές περιβάλλοντος!")

# Αρχικοποίηση Gemini Client
ai_client = genai.Client(api_key=GEMINI_API_KEY)

# Intents για το Discord Bot
intents = discord.Intents.default()
intents.message_content = True  # Απαραίτητο για την ανάγνωση περιεχομένου μηνυμάτων

bot = commands.Bot(command_prefix="!", intents=intents)

# Προαιρετικό: Βάλε το ID του Server σου για ΑΜΕΣΟ sync των εντολών.
# (Αν το αφήσεις None, θα κάνει μόνο Global sync που μπορεί να καθυστερήσει έως 1 ώρα)
GUILD_ID = None  # Π.χ. discord.Object(id=123456789012345678)


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user.name} (ID: {bot.user.id})")
    
    try:
        if GUILD_ID:
            bot.tree.copy_global_to(guild=GUILD_ID)
            synced = await bot.tree.sync(guild=GUILD_ID)
            print(f"Synced {len(synced)} command(s) directly to Guild ID {GUILD_ID.id}")
        else:
            synced = await bot.tree.sync()
            print(f"Synced {len(synced)} command(s) globally.")
    except Exception as e:
        print(f"Sync error: {e}")


@bot.tree.command(
    name="tldr",
    description="Δημιουργεί σύνοψη (TL;DR) των μηνυμάτων του καναλιού.",
)
@app_commands.describe(hours="Πόσες ώρες πίσω να ανατρέξει το bot (1 έως 72)")
async def tldr(interaction: discord.Interaction, hours: int):
    # Ενημέρωση του Discord ότι η επεξεργασία ξεκίνησε
    await interaction.response.defer(thinking=True)

    # Έλεγχος ορίων ωρών
    if hours <= 0 or hours > 72:
        await interaction.followup.send("Παρακαλώ δώσε έναν αριθμό ωρών μεταξύ 1 και 72.")
        return

    # Υπολογισμός χρονικού ορίου
    cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
    messages_list = []

    # Ανάγνωση ιστορικού μηνυμάτων
    async for message in interaction.channel.history(after=cutoff_time, limit=500):
        # Αγνοούμε bots και κενά μηνύματα
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

    # Σύνταξη του κειμένου συνομιλίας
    chat_log = "\n".join(messages_list)
    
    # Προστασία μεγέθους κειμένου αν υπάρχουν πάρα πολλά μηνύματα
    if len(chat_log) > 15000:
        chat_log = chat_log[-15000:]

    # Prompt για το Gemini
    prompt = f"""
Είσαι ένας βοηθός Discord bot. Παρακάτω είναι μια συνομιλία από ένα Discord κανάλι που περιέχει Ελληνικά, Greeklish και Αγγλικά.

Στόχος σου είναι να φτιάξεις ένα καθαρό, δομημένο TL;DR (σύνοψη) στα Ελληνικά για να καταλάβει ο χρήστης τι έχασε όσο έλειπε.

Οδηγίες:
- Ομαδοποίησε τα κύρια θέματα συζήτησης σε bullet points.
- Αναέφερε ποιοι χρήστες συμμετείχαν στα βασικά θέματα αν είναι σχετικό.
- Αν υπήρχαν σημαντικές αποφάσεις, links ή ανακοινώσεις, σημείωσέ τες.
- Κράτα το ύφος φιλικό και σύντομο.

Ιστορικό Συνομιλίας:
{chat_log}
"""

    try:
        # Κλήση του Gemini API με το σωστό μοντέλο
        response = ai_client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
        )
        summary = response.text

        # Διαχείριση ορίου 2000 χαρακτήρων του Discord
        header = f"**TL;DR Τελευταίων {hours} Ωρών** 📝\n\n"
        if len(header + summary) > 2000:
            summary = (
                summary[: 1900 - len(header)]
                + "...\n*(Η σύνοψη κόπηκε λόγω ορίου χαρακτήρων)*"
            )

        await interaction.followup.send(header + summary)

    except Exception as e:
        print(f"DETAILED GEMINI ERROR: {type(e).__name__}: {e}")
        await interaction.followup.send(
            f"Υπήρξε σφάλμα κατά τη επικοινωνία με το AI API: `{e}`"
        )


# Εκκίνηση του Bot
bot.run(DISCORD_TOKEN)
