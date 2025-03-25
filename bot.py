import discord
from discord.ext.commands import Bot

from secrets import DISCORD_TOKEN

intents = discord.Intents.all()
bot = Bot(intents=intents, command_prefix=lambda b, _: b.user.mention)

@bot.event
async def on_ready():
    await bot.load_extension("dream")
    await bot.tree.sync()
    print("Ready")

# Run the bot
bot.run(DISCORD_TOKEN)
