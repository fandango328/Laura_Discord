import io
import base64
import aiohttp
import asyncio
import discord
import calendar
from datetime import datetime, timedelta
from PIL import Image
from discord import app_commands
from discord.ext import commands
from typing import Optional, Literal, Tuple, Coroutine

from constants import (ADETAILER_ARGS, REGIONAL_PROMPTER_ARGS, API_ENDPOINT, LOADING_EMOJI, IMAGE_COOLDOWN,
                       CHECKPOINT_CHOICES, UPSCALER_CHOICES, RESOLUTION_CHOICES, PDXL_CHECKPOINT_CHOICES,
                       VIP_USER_IDS, MAX_IMG2IMG_SIZE, PDXL_ORIENTATION_CHOICES, ILLUSTRIOUS_CHECKPOINTS,)

def scale_to_size(width: int, height: int, size: int) -> Tuple[int, int]:
    scale = (size / (width * height)) ** 0.5
    return int(width * scale), int(height * scale)


class DreamCog(commands.Cog):
    def __init__(self, bot: discord.Client):
        self.bot = bot
        self.queue: list[Tuple[Coroutine, discord.Interaction]] = []
        self.queue_task: Optional[asyncio.Task] = None
        self.generating: dict[int, bool] = {}
        self.last_img: dict[int, datetime] = {}

    def queue_add(self, interaction: discord.Interaction, payload: dict):
        print(f"{interaction.user.name} added to the queue")
        self.generating[interaction.user.id] = True
        self.queue.append((self.fulfill_request(interaction, payload), interaction))
        if not self.queue_task or self.queue_task.done():
            self.queue_task = asyncio.create_task(self.consume_queue())

    async def consume_queue(self):
        new = True
        while self.queue:
            task, interaction = self.queue.pop(0)
            alive = True
            if not new:
                try:
                    await interaction.edit_original_response(content=f"{LOADING_EMOJI} `Generating image...`")
                except discord.errors.NotFound:
                    self.generating[interaction.user.id] = False
                    alive = False
                except Exception as e:
                    print(f"Editing message in queue: {e}")
            if self.queue:
                asyncio.create_task(self.edit_queue_messages())
            if alive:
                await task
            new = False

    async def edit_queue_messages(self):
        tasks = [ctx.edit_original_response(content=f"{LOADING_EMOJI} `Position in queue: {i + 1}`")
                 for i, (task, ctx) in enumerate(self.queue)]
        await asyncio.gather(*tasks, return_exceptions=True)

    def get_loading_message(self):
        if self.queue_task and not self.queue_task.done():
            return f"{LOADING_EMOJI} `Position in queue: {len(self.queue) + 1}`"
        else:
            return f"{LOADING_EMOJI} `Generating image...`"

    async def fulfill_request(self, interaction: discord.Interaction, payload: dict):
        if "init_images" in payload:
            url = API_ENDPOINT + "img2img"
        else:
            url = API_ENDPOINT + "txt2img"

        # Contact the webui api
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        response.raise_for_status()
                    data = await response.json()
        except aiohttp.ClientConnectorError:
            await interaction.edit_original_response(content="Timed out! The AI server is offline.")
        except Exception as e:
            print(f"Generating image: {e}")
            content = "There was a problem getting your image. Please contact the bot owner for details."
            await interaction.edit_original_response(content=content)
        finally:
            # Reset cooldown whether it was a success or fail
            self.generating[interaction.user.id] = False
            self.last_img[interaction.user.id] = datetime.utcnow()

        # Save images to memory and send them
        files = []
        for i in range(len(data["images"])):
            image_data = base64.b64decode(data["images"][i])
            image_stream = io.BytesIO(image_data)
            file = discord.File(image_stream, filename=f"image{i+1}.png")
            files.append(file)
        await interaction.edit_original_response(content="", attachments=files)

    # Commands go after this point

    @app_commands.command(name="dream", description="Generates an AI image using Stable Diffusion.")
    @app_commands.choices(checkpoint=CHECKPOINT_CHOICES, orientation=RESOLUTION_CHOICES, upscaler=UPSCALER_CHOICES)
    @app_commands.describe(prompt="What you want to generate.",
                           checkpoint="The AI model you want to use.",
                           orientation="The resolution of your image.",
                           upscaler="The AI upscaler you want to use.")
    async def dream_command(
            self,
            interaction: discord.Interaction,
            prompt: str,
            checkpoint: str,
            orientation: str,
            upscaler: str
    ):
        # Check cooldown
        if interaction.user.id not in VIP_USER_IDS:
            if self.generating.get(interaction.user.id, False):
                content = "Your current image must finish generating before you can request another one."
                return await interaction.response.send_message(content=content, ephemeral=True)
            if interaction.user.id in self.last_img and (datetime.utcnow() - self.last_img[interaction.user.id]).seconds < IMAGE_COOLDOWN:
                eta = self.last_img[interaction.user.id] + timedelta(seconds=IMAGE_COOLDOWN)
                content = f"You may use this command again <t:{calendar.timegm(eta.utctimetuple())}:R>."
                return await interaction.response.send_message(content=content, ephemeral=True)

        payload = {
            "prompt": f"masterpiece, best quality, {prompt}",
            "negative_prompt": "(worst quality, low quality:2), interlocked fingers, badly drawn hands and fingers, anatomically incorrect hands,",
            "sampler_name": "DPM++ 2M Karras",
            "steps": 26,
            "cfg_scale": 6.5,
            "denoising_strength": 0.40,
            "width": int(orientation.split("x")[0]),
            "height": int(orientation.split("x")[1]),
            "override_settings": {
                "sd_model_checkpoint": checkpoint,
                "sd_vae": "vae-ft-mse-840000-ema-pruned.safetensors",
            },
            "override_settings_restore_afterwards": "True",
            "enable_hr": "True",
            "hr_scale": 2,
            "hr_upscaler": upscaler,
            "hr_second_pass_steps": "14",
            "alwayson_scripts": {
                "ADetailer": ADETAILER_ARGS
            }
        }

        # Add task to the queue
        content = self.get_loading_message()
        self.queue_add(interaction, payload)
        await interaction.response.send_message(content=content)

    @app_commands.command(name="redream", description="Converts an existing image using Stable Diffusion AI.")
    @app_commands.choices(checkpoint=CHECKPOINT_CHOICES, orientation=RESOLUTION_CHOICES)
    @app_commands.describe(image="The image you want the AI to use as a base.",
                           prompt="What you want to generate.",
                           checkpoint="The AI model you want to use.",
                           orientation="The resolution of your image.",
                           denoising="How much you want the image to change. Try 0.6")
    async def img2img_command(
            self,
            interaction: discord.Interaction,
            image: discord.Attachment,
            prompt: str,
            checkpoint: str,
            orientation: str,
            denoising: app_commands.Range[float, 0.0, 1.0],
    ):
        # Check cooldown
        if interaction.user.id not in VIP_USER_IDS:
            if self.generating.get(interaction.user.id, False):
                content = "Your current image must finish generating before you can request another one."
                return await interaction.response.send_message(content=content, ephemeral=True)
            if interaction.user.id in self.last_img and (
                    datetime.utcnow() - self.last_img[interaction.user.id]).seconds < IMAGE_COOLDOWN:
                eta = self.last_img[interaction.user.id] + timedelta(seconds=IMAGE_COOLDOWN)
                content = f"You may use this command again <t:{calendar.timegm(eta.utctimetuple())}:R>."
                return await interaction.response.send_message(content=content, ephemeral=True)

        if not image.content_type.startswith("image/"):
            return await interaction.response.send_message("The file you uploaded is not a valid image.", ephemeral=True)

        # Make Discord wait while we download the image
        await interaction.response.defer()

        # Save and resize the image if necessary
        fp = io.BytesIO()
        await image.save(fp)
        if image.width * image.height > MAX_IMG2IMG_SIZE:
            width, height = scale_to_size(image.width, image.height, MAX_IMG2IMG_SIZE)
            resized_image = Image.open(fp).resize((width, height), Image.Resampling.LANCZOS)
            fp = io.BytesIO()
            resized_image.save(fp, "PNG")
            fp.seek(0)
        encoded_image = base64.b64encode(fp.read()).decode("utf8")

        payload = {
            "prompt": f"masterpiece, best quality, {prompt}",
            "negative_prompt": "(worst quality, low quality:2), interlocked fingers, badly drawn hands and fingers, anatomically incorrect hands,",
            "sampler_name": "DPM++ 2M Karras",
            "steps": 30,
            "cfg_scale": 6.5,
            "width": int(orientation.split("x")[0]),
            "height": int(orientation.split("x")[1]),
            "override_settings": {
                "sd_model_checkpoint": checkpoint
            },
            "override_settings_restore_afterwards": "True",
            "init_images": [encoded_image],
            "denoising_strength": denoising,
            "alwayson_scripts": {
                "ADetailer": ADETAILER_ARGS
            }
        }

        # Add task to the queue
        content = self.get_loading_message()
        self.queue_add(interaction, payload)
        await interaction.edit_original_response(content=content)

    @app_commands.command(name="catnap", description="Generates an image with preset options")
    @app_commands.choices(orientation=PDXL_ORIENTATION_CHOICES)
    async def catnap_command(
            self,
            interaction: discord.Interaction,
            preset: Literal["Atomix", "Retro", "Illustrious", "Laura", "Unholy"],
            #checkpoint: str,
            orientation: str,
            prompt: str,
            neg_prompt: str,
            batch_size: Literal[1, 2, 4]
    ):
        # Check cooldown
        if interaction.user.id not in VIP_USER_IDS:
            if self.generating.get(interaction.user.id, False):
                content = "Your current image must finish generating before you can request another one."
                return await interaction.response.send_message(content, ephemeral=True)
            if interaction.user.id in self.last_img and (datetime.utcnow() - self.last_img[interaction.user.id]).seconds < IMAGE_COOLDOWN:
                eta = self.last_img[interaction.user.id] + timedelta(seconds=IMAGE_COOLDOWN)
                content = f"You may use this command again <t:{calendar.timegm(eta.utctimetuple())}:R>."
                return await interaction.response.send_message(content, ephemeral=True)
   
        if preset == "Retro":
            payload = {
                "prompt": f"masterpiece,best quality, highly detailed, score_9, score_8_up, score_7_up, score_6_up, {prompt}",
                "negative_prompt": "3d, monochrome, simple background,watermark, patreon username, artist name, signature, text",
                "sampler_name": "Euler A",
                "steps": 26,
                "cfg_scale": 6,
                "denoising_strength": 0.44,
                "width": int(orientation.split("x")[0]),
                "height": int(orientation.split("x")[1]),
                "override_settings": {
                    "sd_model_checkpoint": "pottasticpdxl_"
                },
                "override_settings_restore_afterwards": "True",
                "enable_hr": "True",
                "hr_scale": 2,
                "hr_upscaler": "4x-AnimeSharp",
                "hr_second_pass_steps": "14",
                "alwayson_scripts": {
                    "ADetailer": ADETAILER_ARGS
                }
            }
        elif preset == "realistic":
            payload = {
                "prompt": f"score_9, score_8_up, score_7_up, score_6_up, score_5_up, score_4_up, realistic {prompt}",
                "negative_prompt": "(score_1, score_2, score_3), sketch, worst quality, low quality, deformed, censored, bad anatomy, patreon, logo, ",
                "sampler_name": "DPM++ 2M SDE SGMUniform",
                "steps": 20,
                "cfg_scale": 4,
                "denoising_strength": 0.15,
                "width": int(orientation.split("x")[0]),
                "height": int(orientation.split("x")[1]),
                "override_settings": {
                    "sd_model_checkpoint": "2dnPony_v10Play"
                },
                "override_settings_restore_afterwards": "True",
                "enable_hr": "True",
                "hr_scale": 2,
                "hr_upscaler": "4xRealWebPhoto_v4_dat2",
                "hr_second_pass_steps": "10",
                "alwayson_scripts": {
                    "ADetailer": ADETAILER_ARGS
                }
            }
        elif preset == "Unholy":
            payload = {
                "prompt": f"masterpiece, best quality, {prompt}",
                "negative_prompt": f"worst quality, low quality, lowres, jpeg artifacts, bad anatomy, bad hands, watermark, {neg_prompt}",
                "sampler_name": "Euler a",
                "batch_size": batch_size,
                "steps": 28,
                "denoising_strength": 0.15,
                "width": int(orientation.split("x")[0]),
                "height": int(orientation.split("x")[1]),
                "override_settings": {
                    "sd_model_checkpoint": "holyMixIllustriousxl_v1"
                },
                "override_settings_restore_afterwards": "True",
                "enable_hr": "False",
                "hr_scale": 2,
                "hr_upscaler": "4x-UltraSharp",
                "hr_second_pass_steps": "10",
                "alwayson_scripts": {
                    "ADetailer": ADETAILER_ARGS
                }
            }       
        elif preset == "Laura":
            payload = {
                "prompt": f"masterpiece, best quality, {prompt}",
                "negative_prompt": f"worst quality, low quality, bad anatomy, watermark, username, patreon, {neg_prompt}",
                "sampler_name": "Euler a",
                "batch_size": batch_size,
                "steps": 24,
                "denoising_strength": 0.15,
                "width": int(orientation.split("x")[0]),
                "height": int(orientation.split("x")[1]),
                "override_settings": {
                    "sd_model_checkpoint": "pasanctuarySDXL_v40"
                },
                "override_settings_restore_afterwards": "True",
                "enable_hr": "False",
                "hr_scale": 2,
                "hr_upscaler": "4x-UltraSharp",
                "hr_second_pass_steps": "10",
                "alwayson_scripts": {
                    "ADetailer": ADETAILER_ARGS
                }
            }
        elif preset == "Illustrious":
            payload = {
                "prompt": f"masterpiece, best quality, {prompt}",
                "negative_prompt": f"worst quality, low quality, bad anatomy, watermark, username, patreon, {neg_prompt}",
                "sampler_name": "Euler a",
                "batch_size": batch_size,
                "steps": 24,
                "denoising_strength": 0.15,
                "width": int(orientation.split("x")[0]),
                "height": int(orientation.split("x")[1]),
                "override_settings": {
                    "sd_model_checkpoint": "pasanctuarySDXL_v40"
                },
                "override_settings_restore_afterwards": "True",
                "enable_hr": "True",
                "hr_scale": 2,
                "hr_upscaler": "4x-UltraSharp",
                "hr_second_pass_steps": "10",
                "alwayson_scripts": {
                    "ADetailer": ADETAILER_ARGS
                }
            }
        elif preset == "Atomix":
            payload = {
                "prompt": f"score_8, score_6_up, 8k RAW photo, film grain, realistic, {prompt}",
                "negative_prompt": "(worst quality, low quality:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), patreon, logo,",
                "sampler_name": "DPM++ 2M SDE SGMUniform",
                "steps": 14,
                "cfg_scale": 2,
                "denoising_strength": 0.44,
                "width": int(orientation.split("x")[0]),
                "height": int(orientation.split("x")[1]),
                "override_settings": {
                    "sd_model_checkpoint": "atomixPonyRealismXL_v10"
                },
                "override_settings_restore_afterwards": "True",
                "enable_hr": "False",
                "hr_scale": 2,
                "hr_upscaler": "4x-UltraSharp",
                "hr_second_pass_steps": "18",
                "alwayson_scripts": {
                    "ADetailer": ADETAILER_ARGS
                }
            }
        elif preset == "PDXL":
            payload = {
                "prompt": f"{prompt}",
                "negative_prompt": "greyscale,simple background,3d,blurry,monochrome,text,watermark,nose,patreon",
                "sampler_name": "Euler a",
                "batch_size": batch_size,
                "steps": 26,
                "cfg_scale": 5.5,
                "denoising_strength": 0.44,
                "width": int(orientation.split("x")[0]),
                "height": int(orientation.split("x")[1]),
                "override_settings": {
                "sd_model_checkpoint": checkpoint,
                "sd_vae": "sdxl_vae.safetensors",
                },
                "override_settings_restore_afterwards": True,  # Removed quotes from True
                "enable_hr": False,  # Removed quotes from True
                "hr_scale": 1.5,
                "hr_upscaler": "4x-AnimeSharp",
                "hr_second_pass_steps": 12,  # Removed quotes from number
                "alwayson_scripts": {
                    "ADetailer": ADETAILER_ARGS
                }
            }
        # Add task to the queue
        content = self.get_loading_message()
        self.queue_add(interaction, payload)
        await interaction.response.send_message(content=content)

    @app_commands.command(name="ephemeral", description="Next level dreams...")
    @app_commands.choices(orientation=PDXL_ORIENTATION_CHOICES)
    async def ephemeral_command(
            self,
            interaction: discord.Interaction,
            orientation: str,
            common: str,
            left_side: str,
            right_side: str,
            batch_size: Literal["1",]
    ):
        # Check cooldown
        if interaction.user.id not in VIP_USER_IDS:
            if self.generating.get(interaction.user.id, False):
                content = "Your current image must finish generating before you can request another one."
                return await interaction.response.send_message(content, ephemeral=True)
            if interaction.user.id in self.last_img and (datetime.utcnow() - self.last_img[interaction.user.id]).seconds < IMAGE_COOLDOWN:
                eta = self.last_img[interaction.user.id] + timedelta(seconds=IMAGE_COOLDOWN)
                content = f"You may use this command again <t:{calendar.timegm(eta.utctimetuple())}:R>."
                return await interaction.response.send_message(content, ephemeral=True)

        payload = {
                "prompt": f"masterpiece, best quality, {common} ADDCOMM {left_side} ADDCOL {right_side}",
                "negative_prompt": "worst quality, low quality, bad anatomy, watermark, username, patreon,",
                "sampler_name": "Euler a",
                "batch_size": batch_size,
                "steps": 26,
                "cfg_scale": 6,
                "denoising_strength": 0.20,
                "width": int(orientation.split("x")[0]),
                "height": int(orientation.split("x")[1]),
                "override_settings": {
                    "sd_model_checkpoint": "holyMixIllustriousxl_v1",
                    "sd_vae": "sdxl_vae.safetensors",
                },
                "override_settings_restore_afterwards": True,
                "enable_hr": True,
                "hr_scale": 1.5,
                "hr_upscaler": "4x-UltraSharp",
                "hr_second_pass_steps": 10,
                "alwayson_scripts": {
                    "Regional Prompter": REGIONAL_PROMPTER_ARGS,
                    "ADetailer": ADETAILER_ARGS
                }
            }
  
        # Add task to the queue
        content = self.get_loading_message()
        self.queue_add(interaction, payload)
        await interaction.response.send_message(content=content)

# Add this module to the bot
async def setup(bot):
    await bot.add_cog(DreamCog(bot))
