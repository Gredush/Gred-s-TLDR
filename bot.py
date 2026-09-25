import asyncio
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

GUILD_ID = None  # Βάλε discord.Object(id=...) αν θέλεις άμεσο sync


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


@bot.tree.command(
    name="tldr",
    description="Δημιουργεί σύνοψη (TL;DR) των μηνυμάτων του καναλιού.",
)
@app_commands.describe(hours="Πόσες ώρες πίσω να ανατρέξει το bot (1 έως 72)")
async def tldr(interaction: discord.Interaction, hours: int):
    await interaction.response.defer(thinking=True)

    if hours <= 0 or hours > 72:
        await interaction.followup.send(
            "Παρακαλώ δώσε έναν αριθμό ωρών μεταξύ 1 και 72."
        )
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

    if len(chat_log) > 15000:
        chat_log = chat_log[-15000:]

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

    # Λίστα με διαθέσιμα μοντέλα κατά σειρά προτίμησης
    candidate_models = ["gemini-2.5-flash", "gemini-1.5-flash"]

    summary = None
    last_error = None

    for model_name in candidate_models:
        try:
            # Δοκιμή κλήσης στο μοντέλο
            response = ai_client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            summary = response.text
            if summary:
                break  # Αν πετύχει, βγαίνουμε από το loop
        except Exception as e:
            last_error = e
            print(
                f"Model {model_name} failed with error: {e}. Trying next"
                " model..."
            )
            await asyncio.sleep(1)  # Μικρή αναμονή πριν τη επόμενη δοκιμή

    if not summary:
        await interaction.followup.send(
            "Το API της Google είναι προσωρινά υπερφορτωμένο (503 High Demand)."
            " Παρακαλώ δοκίμασε ξανά σε 1-2 λεπτά."
        )
        return

    header = f"**TL;DR Τελευταίων {hours} Ωρών** 📝\n\n"
    if len(header + summary) > 2000:
        summary = (
            summary[: 1900 - len(header)]
            + "...\n*(Η σύνοψη κόπηκε λόγω ορίου χαρακτήρων)*"
        )

    await interaction.followup.send(header + summary)


bot.run(DISCORD_TOKEN)
