import os
from datetime import datetime, timedelta, timezone
import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv

# API SDKs
from google import genai
from groq import AsyncGroq
from openai import AsyncOpenAI

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Αρχικοποίηση Clients
groq_client = AsyncGroq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
openrouter_client = (
    AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
    if OPENROUTER_API_KEY
    else None
)

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)
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


async def generate_summary_with_fallback(system_prompt: str, chat_log: str) -> str:
    """
    Δοκιμάζει διαδοχικά Providers και Models μέχρι να βρει ένα που απαντάει.
    """
    providers = []

    # 1. Groq Provider
    if groq_client:
        providers.append(
            ("Groq", groq_client, ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"])
        )

    # 2. Gemini Provider
    if gemini_client:
        providers.append(
            ("Gemini", gemini_client, ["gemini-2.0-flash", "gemini-1.5-flash"])
        )

    # 3. OpenRouter Provider (Free Tier Backup)
    if openrouter_client:
        providers.append(
            (
                "OpenRouter",
                openrouter_client,
                [
                    "google/gemini-2.0-flash-exp:free",
                    "meta-llama/llama-3.3-70b-instruct:free",
                ],
            )
        )

    last_error = "Δεν βρέθηκε διαθέσιμο API Key."

    for provider_name, client, models in providers:
        for model_name in models:
            try:
                print(f"Trying provider: {provider_name} | Model: {model_name}...")

                if provider_name == "Groq":
                    response = await client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"Ιστορικό Συνομιλίας:\n{chat_log}"},
                        ],
                        temperature=0.5,
                        max_tokens=1000,
                    )
                    summary = response.choices[0].message.content

                elif provider_name == "Gemini":
                    full_prompt = f"{system_prompt}\n\nΙστορικό Συνομιλίας:\n{chat_log}"
                    response = await client.aio.models.generate_content(
                        model=model_name,
                        contents=full_prompt,
                    )
                    summary = response.text

                elif provider_name == "OpenRouter":
                    response = await client.chat.completions.create(
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
                    print(f"Success with {provider_name} ({model_name})!")
                    return summary

            except Exception as e:
                last_error = f"{provider_name} ({model_name}): {e}"
                print(f"Failed {provider_name} ({model_name}) -> {e}")

    raise RuntimeError(f"Όλα τα AI APIs απέτυχαν. Τελευταίο σφάλμα: {last_error}")


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

    try:
        summary = await generate_summary_with_fallback(system_prompt, chat_log)

        header = f"**TL;DR Τελευταίων {hours} Ωρών** 📝\n\n"
        if len(header + summary) > 2000:
            summary = (
                summary[: 1900 - len(header)]
                + "...\n*(Η σύνοψη κόπηκε λόγω ορίου χαρακτήρων)*"
            )

        await interaction.followup.send(header + summary)

    except Exception as e:
        print(f"Fallback Chain Exhausted: {e}")
        await interaction.followup.send(
            f"Υπήρξε πρόβλημα με όλες τις υπηρεσίες AI: `{e}`"
        )


bot.run(DISCORD_TOKEN)
