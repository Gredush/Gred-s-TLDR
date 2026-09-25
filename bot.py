import os
from datetime import datetime, timedelta, timezone
import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv

import aiohttp
from google import genai
from groq import AsyncGroq

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Αρχικοποίηση SDK Clients
groq_client = AsyncGroq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

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


async def call_gemini_rest_fallback(prompt: str) -> str:
    """
    Direct REST HTTP Call στο Gemini API χρησιμοποιώντας το ενεργό gemini-3.8-flash.
    """
    # Ενημερωμένο μοντέλο σε gemini-3.8-flash
    models_to_try = ["gemini-3.8-flash", "gemini-1.5-flash"]
    
    async with aiohttp.ClientSession() as session:
        for model in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}]
            }
            try:
                async with session.post(url, json=payload, timeout=30) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data['candidates'][0]['content']['parts'][0]['text']
                    else:
                        err_text = await resp.text()
                        print(f"[Gemini REST Warning] {model} returned {resp.status}: {err_text}")
            except Exception as e:
                print(f"[Gemini REST Exception] {model}: {e}")
                
    raise RuntimeError("Όλα τα Gemini REST endpoints επέστρεψαν σφάλμα.")


async def generate_summary_with_fallback(system_prompt: str, chat_log: str) -> str:
    last_error = None
    full_prompt = f"{system_prompt}\n\nΙστορικό Συνομιλίας:\n{chat_log}"

    # ---------------------------------------------------------
    # 1. GROQ SDK
    # ---------------------------------------------------------
    if groq_client:
        groq_models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
        for model_name in groq_models:
            try:
                print(f"[Groq SDK] Δοκιμή με {model_name}...")
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
                    print(f"[Groq SDK SUCCESS] {model_name}")
                    return summary
            except Exception as e:
                last_error = f"Groq SDK ({model_name}): {e}"
                print(f"[Groq SDK ERROR] {last_error}")

    # ---------------------------------------------------------
    # 2. GEMINI SDK (gemini-3.8-flash)
    # ---------------------------------------------------------
    if gemini_client:
        gemini_models = ["gemini-3.8-flash", "gemini-1.5-flash"]
        for model_name in gemini_models:
            try:
                print(f"[Gemini SDK] Δοκιμή με {model_name}...")
                response = await gemini_client.aio.models.generate_content(
                    model=model_name,
                    contents=full_prompt,
                )
                summary = response.text
                if summary and summary.strip():
                    print(f"[Gemini SDK SUCCESS] {model_name}")
                    return summary
            except Exception as e:
                last_error = f"Gemini SDK ({model_name}): {e}"
                print(f"[Gemini SDK ERROR] {last_error}")

    # ---------------------------------------------------------
    # 3. GEMINI REST API FALLBACK
    # ---------------------------------------------------------
    if GEMINI_API_KEY:
        try:
            print("[Gemini REST] Δοκιμή απευθείας HTTP κλήσης...")
            summary = await call_gemini_rest_fallback(full_prompt)
            if summary and summary.strip():
                print("[Gemini REST SUCCESS]")
                return summary
        except Exception as e:
            last_error = f"Gemini REST: {e}"
            print(f"[Gemini REST ERROR] {last_error}")

    raise RuntimeError(f"Όλα τα AI APIs απέτυχαν. Τελευταίο καταγεγραμμένο σφάλμα: {last_error}")

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
        " points. Αναφέρε ποιοι χρήστες συμμετείχαν στα βασικά θέματα και αν"
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
            f"Υπήρξε πρόβλημα με τις υπηρεσίες AI: `{e}`"
        )


bot.run(DISCORD_TOKEN)
