import os
from datetime import datetime, timedelta, timezone
import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv
from groq import AsyncGroq

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not DISCORD_TOKEN or not GROQ_API_KEY:
    print("CRITICAL ERROR: Λείπουν τα API Keys (DISCORD_TOKEN ή GROQ_API_KEY)!")

# Αρχικοποίηση Async Groq Client
groq_client = AsyncGroq(api_key=GROQ_API_KEY)

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

# Βάλε το ID του server σου αν θέλεις ακαριαίο sync (π.χ. discord.Object(id=123456789))
GUILD_ID = None


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

    system_prompt = (
        "Είσαι ένας βοηθός Discord bot. Η δουλειά σου είναι να διαβάζεις"
        " συνομιλίες (που περιέχουν Ελληνικά, Greeklish και Αγγλικά) και να"
        " φτιάχνεις μια καθαρή, δομημένη σύνοψη (TL;DR) στα Ελληνικά με bullet"
        " points. Αναέφερε ποιοι χρήστες συμμετείχαν στα βασικά θέματα και αν"
        " υπήρχαν σημαντικές αποφάσεις ή links."
    )

    # Μόνο τα ενεργά και έγκυρα μοντέλα του Groq
    candidate_models = ["llama-3.1-8b-instant", "llama-3.3-70b-specdec"]

    summary = None
    last_error = None

    for model_name in candidate_models:
        try:
            response = await groq_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"Ιστορικό Συνομιλίας:\n{chat_log}",
                    },
                ],
                temperature=0.5,
                max_tokens=1000,
            )
            summary = response.choices[0].message.content
            if summary:
                break
        except Exception as e:
            last_error = e
            print(f"Groq model {model_name} failed: {e}")

    if not summary:
        await interaction.followup.send(
            f"Υπήρξε σφάλμα κατά τη επικοινωνία με το AI API: `{last_error}`"
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
