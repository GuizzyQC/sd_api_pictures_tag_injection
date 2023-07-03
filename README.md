## Description:
TL;DR: Lets the bot answer you with a picture!  

Stable Diffusion API pictures for TextGen with Tag Injection, v.1.0.0  
Based on [Brawlence's extension](https://github.com/Brawlence/SD_api_pics) to [oobabooga's textgen-webui](https://github.com/oobabooga/text-generation-webui) allowing you to receive pics generated by [Automatic1111's SD-WebUI API](https://github.com/AUTOMATIC1111/stable-diffusion-webui). Including improvements from ClayShoaf.

This extension greatly improves usability of the sd_api_extension in chat mode, especially for RP scenarios. It allows a character's appearance that has been crafted in Automatic1111's UI to be copied into the character sheet and then inserted dynamically into the SD prompt when the text-generation-webui extension sees the character has been asked to send a picture of itself, allowing the same finely crafted SD tags to be send each time, including LORAs if they were used. It also allows for extra SD tags to be added if the input prompt or the character's response contains strings defined in the translations.json file. Check the examples below for ideas how to use this.

## Installation

To install, in a command line, navigate to your text-generation-webui folder, then enter the extensions folder and then `git clone https://github.com/GuizzyQC/sd_api_pictures_tag_injection.git`

## Usage

Load it in the `--chat` mode with `--extension sd_api_pictures_tag_injection`.  

The image generation is triggered either:  
- manually through the 'Force the picture response' button while in `Manual` or `Immersive/Interactive` modes OR  
- automatically in `Immersive/Interactive` mode if the words `'send|main|message|me'` are followed by `'image|pic|picture|photo|snap|snapshot|selfie|meme'` in the user's prompt  
- always on in Picturebook/Adventure mode (if not currently suppressed by 'Suppress the picture response')  

## Prerequisites

One needs an available instance of Automatic1111's webui running with an `--api` flag. Ain't tested with a notebook / cloud hosted one but should be possible.   
To run it locally in parallel on the same machine, specify custom `--listen-port` for either Auto1111's or ooba's webUIs.  

## Features:
- Dynamic injection of content into SD prompt upon detection of a preset "translation" string  
- Dynamic injection of content into SD prompt upon detection of a request for character selfie  
- Dynamic injection of content into SD prompt upon detection of a specific checkpoint being selected  
- SD Checkpoint selection
- Secondary tags injection, useful to toggle between two styles of images without having to manually type in tags  
- API detection (press enter in the API box)  
- VRAM management (model shuffling)  
- Three different operation modes (manual, interactive, always-on)  
- persistent settings via settings.json

The model input is modified only in the interactive mode; other two are unaffected. The output pic description is presented differently for Picture-book / Adventure mode.  

### Checkpoint file Stable Diffusion tags

If the "Add checkpoint tags in prompt" option is selected, if the checkpoint you loaded matches one in the checkpoints.json file it will add the relevant tags to your prompt. The format for the checkpoints.json file is as follow:

JSON:
```json
{
	"pairs": [{
		"name": "toonyou_beta3.safetensors [52768d2bc4]",
		"positive_prompt": "cartoon",
		"negative_prompt": "photograph, realistic"
	        },
		{"name": "analogMadness_v50.safetensors [f968fc436a]",
		"positive_prompt": "photorealistic, realistic",
		"negative_prompt": "cartoon, render, anime"
	    }]
}
```

### Character sheet Stable Diffusion tags

In immersive mode, to help your character maintain a better fixed image, add positive_sd and negative_sd to your character's json file to have Stable Diffusion tags that define their appearance automatically added to Stable Diffusion prompts whenever the extension detects the character was asked to send a picture of itself, ex:

JSON:
```json
{
	"sd_tags_positive": "24 year old, asian, long blond hair, ((twintail)), blue eyes, soft skin, height 5'8, woman, <lora:shojovibe_v11:0.1>",
	"sd_tags_negative": "old, elderly, child, deformed, cross-eyed"
}
```

YAML:
```yaml
sd_tags_positive: 24 year old, asian, long blond hair, ((twintail)), blue eyes, soft
  skin, height 5'8, woman, <lora:shojovibe_v11:0.1>
sd_tags_negative: old, elderly, child, deformed, cross-eyed
```

If clothing and accessories are permanently affixed and will never be changed in any picture you will request of that character, feel free to add them to this tag too. The extension prompts the character to describe what it is wearing whenever a picture of itself is requested as to keep that aspect dynamic, so adding it in the character json makes it more static.

A good sample prompt to trigger this is "Send me a picture of you", followed or not by more details about requested action and context.


### Description to Stable Diffusion translation

Whenever the Activate SD translations box is checked, the extension will load the translations.json file when a picture is requested, and will check in both the request to the language model, as well as the response of the language model, for specific words listed in the translations.json file and will add words or tags to the Stable Diffusion prompt accordingly, ex:

JSON:
```json
{
	"pairs": [{
		"descriptive_word": ["tennis"],
		"SD_positive_translation": "tennis ball, rackets, (net), <lora:povTennisPlaying_lora:0.5>",
		"SD_negative_translation": ""
	        },
		{"descriptive_word": ["soccer","football"],
		"SD_positive_translation": "((soccer)), nets",
		"SD_negative_translation": ""
	    }]
}
```

The tags can also include Stable Diffusion LORAs if you have any that are relevant.

#### Character specific translation patterns

If you have translations that you only want to see added for a specific character (for instance, if a specific character has specific clothes or uniforms or physical characteristics that you only want to see triggered when specific words are used), add the translations_patterns heading in your character's JSON or YAML file. The *translations_patterns* heading works exactly the same way as the *pairs* heading does in the translations.json file.

JSON:
```json
  "translation_patterns": [
    {
      "descriptive_word": [
        "tennis"
      ],
      "SD_positive_translation": "cute frilly blue tennis uniform, <lora:frills:0.9>",
      "SD_negative_translation": ""
    },
    {
      "descriptive_word": [
        "basketball",
        "bball"
      ],
      "SD_positive_translation": "blue basketball uniform",
      "SD_negative_translation": "red uniform"
    }
  ]
```

YAML:
```yaml
translation_patterns:
- descriptive_word:
  - tennis
  SD_positive_translation: "cute frilly blue tennis uniform, <lora:frills:0.9>"
  SD_negative_translation: ''
- descriptive_word:
  - basketball
  - bball
  SD_positive_translation: "blue basketball uniform"
  SD_negative_translation: "red uniform"
```

Note that character specific translation patterns stack with the general translation patterns.

### Persistent settings

Create or modify the `settings.json` in the `text-generation-webui` root directory to override the defaults
present in script.py, ex:

JSON:
```json
{
    "sd_api_pictures_tag_injection-manage_VRAM": 1,
    "sd_api_pictures_tag_injection-save_img": 1,
    "sd_api_pictures_tag_injection-prompt_prefix": "(Masterpiece:1.1), detailed, intricate, colorful, (solo:1.1)",
    "sd_api_pictures_tag_injection-secondary_positive_prompt": "<lora:add_details:1.2>",
    "sd_api_pictures_tag_injection-secondary_negative_prompt": "",
    "sd_api_pictures_tag_injection-sampler_name": "DPM++ 2M Karras"
}
```

Will automatically set the `Manage VRAM` & `Keep original images` checkboxes and change the texts in `Prompt Prefix` and `Sampler name` on load as well as setup the seconday positive and negative prompt (without activating them).
