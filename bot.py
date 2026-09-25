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
SUMMARY_CHANNEL_ID = os.getenv("SUMMARY_CHANNEL_ID")  # ID Καναλιού για το αυτόματο TL;DR

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
# AUTO SUMMARY TASK (Κάθε 12 ώρες: 09:00 & 21:00 UTC+3)
# ---------------------------------------------------------
@tasks.loop(minutes=1)
async def daily_summary_task():
    """Ελέγχει κάθε λεπτό αν είναι 09:00 ή 21:00 (UTC+3) για να στείλει το 12ωρο TL;DR."""
    if not SUMMARY_CHANNEL_ID:
        return

    # Ώρα Ελλάδος (UTC+3)
    tz_greece = timezone(timedelta(hours=3))
    now = datetime.now(tz_greece)

    # TriggerΟρίστε το ενημερωμένο σύστημα εντολών με το όριο των 12 ωρών:

### 1. Εντολή `!tldr` (με όριο τις 12 ώρες)

```javascript
// Παράδειγμα ελέγχου ορίου στο !tldr
let hours = parseInt(args[0]) || 12;

// Περιορισμός: Αν ο χρήστης δώσει πάνω από 12 ώρες, το ορίζουμε αυτόματα στο 12
if (hours > 12) {
    hours = 12;
}

// Χρήση της μεταβλητής hours για το fetch των μηνυμάτων...
