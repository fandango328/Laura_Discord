from discord.app_commands import Choice

API_ENDPOINT = "http://127.0.0.1:7860/sdapi/v1/"

IMAGE_COOLDOWN = 15  # Cooldown starts counting after the image is done generating

LOADING_EMOJI = "⌛"  # can replace with an animated server emoji

VIP_USER_IDS = [  # These users can skip the cooldown and make multiple images at the same time
    871733390251012147,
    884988727456329779,
    1236118548661932053,
]

MAX_IMG2IMG_SIZE = 1216 * 1216

CHECKPOINTS = [
    "catCitronAnime_v10",
    "animilfMix_v10",
    "edgNendo_v2AnimePrunedfp16",
    "incursiosMeme_v16Vaefix",
    "luminaverse_v10",
    "helloartdoor_V10b",
    "melange_10f",
    "absolutereality_v16",
    "maturemalemix_v14",
    "anithing_v10",
]
PDXL_CHECKPOINTS = [
    "atomixPonyRealismXL_v10",
    "autismmixSDXL_autismmixPony",
    "incursiosMemeDiffusion_v10",
    "pottasticpdxl_",
    "aaaautismPonyFinetune_v4",
    "xlsusjelomixPony_v10",
    "edgFromZero_v10",
    "perfectPonyXL_v10",
    "richyrichmixpdxl_V1Fp16",
    "mfcgSquishMix_v20",
    "dogepointPDXL_dogeponyREAL"
 ]

ILLUSTRIOUS_CHECKPOINTS = [
    "pasanctuarySDXL_v20",
 ]
FLUX_CHECKPOINTS = [
    "atomixFluxNF4_v10",
]   

UPSCALERS = [
    "4x-AnimeSharp",
    "4x-UltraSharp",
    "4x_foolhardy_Remacri",
]

RESOLUTIONS = {  # Must be two numbers separated by an x
    "Portrait": "512x768",
    "Full-body": "512x910",
    "Landscape": "768x512",
    "Widescreen": "910x512",
    "Square": "512x512",
    "PDXL" : "832x1216",
}

PDXL_ORIENTATION = {  # Must be two numbers separated by an x
    "Portrait": "832x1216",
    "Landscape": "1216x832",
    "Square": "1024x1024"
}

# Don't touch these
CHECKPOINT_CHOICES = [Choice(name=x, value=x) for x in CHECKPOINTS]
PDXL_CHECKPOINT_CHOICES = [Choice(name=x, value=x) for x in PDXL_CHECKPOINTS]
FLUX_CHECKPOINT_CHOICES = [Choice(name=x, value=x) for x in FLUX_CHECKPOINTS]
ILLUSTRIOUS_CHECKPOINT_CHOICES = [Choice(name=x, value=x) for x in ILLUSTRIOUS_CHECKPOINTS]
UPSCALER_CHOICES = [Choice(name=x, value=x) for x in UPSCALERS]
PDXL_ORIENTATION_CHOICES = [Choice(name=k, value=v) for k, v in PDXL_ORIENTATION.items()]
RESOLUTION_CHOICES = [Choice(name=k, value=v) for k, v in RESOLUTIONS.items()]


# Adetailer parameters used on every generated image
ADETAILER_ARGS = {
    "args": [

        {
            "ad_model": "Anzhc Face seg 1024 v2 y8n.pt",
            "ad_prompt": "",
            "ad_negative_prompt": "",
            "ad_confidence": 0.7,
            "ad_mask_k_largest": 0,
            "ad_mask_min_ratio": 0.0,
            "ad_mask_max_ratio": 1.0,
            "ad_dilate_erode": 32,
            "ad_x_offset": 0,
            "ad_y_offset": 0,
            "ad_mask_merge_invert": "None",
            "ad_mask_blur": 24,
            "ad_denoising_strength": 0.30,
            "ad_inpaint_only_masked_padding": 64,
            "ad_inpaint_width": 1024,
            "ad_inpaint_height": 1024,
        },
        #{
        #   "ad_model": "lips_v1.pt",
        #   "ad_prompt": "",
        #   "ad_negative_prompt": "",
        #   "ad_confidence": 0.7,
        #  "ad_mask_k_largest": 0,
        #   "ad_mask_min_ratio": 0.0,
        #   "ad_mask_max_ratio": 1.0,
        #   "ad_dilate_erode": 32,
        #   "ad_x_offset": 0,
        #   "ad_y_offset": 0,
        #   "ad_mask_merge_invert": "None",
        #   "ad_mask_blur": 4,
        #   "ad_denoising_strength": 0.3,
        #   "ad_inpaint_only_masked_padding": 0,
        #   "ad_inpaint_width": 1024,
        #   "ad_inpaint_height": 1024,
        #},
        #{
            #"ad_model": "hand_yolov8n.pt",
            #"ad_prompt": "hand,",
            #"ad_negative_prompt": "",
            #"ad_confidence": 0.7,
            #"ad_mask_k_largest": 0,
            #"ad_mask_min_ratio": 0.0,
            #"ad_mask_max_ratio": 1.0,
            #"ad_dilate_erode": 32,
            #"ad_x_offset": 0,
            #"ad_y_offset": 0,
            #"ad_mask_merge_invert": "None",
            #"ad_mask_blur": 4,
            #"ad_denoising_strength": 0.2,
            #"ad_inpaint_only_masked_padding": 0,
            #"ad_inpaint_width": 1024,
            #"ad_inpaint_height": 1024,
        #},
    ]
}
# Regional Prompter parameters
REGIONAL_PROMPTER_ARGS = {
    "args": [True,False,"Matrix","Vertical","Mask","Prompt","1,1",".2",False,True,False,"Attention",False,"0","0","0","False"]
}
       