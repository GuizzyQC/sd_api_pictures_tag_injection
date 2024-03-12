import base64
import io
import re
import time
from datetime import date
from pathlib import Path

import gradio as gr
import modules.shared as shared
import requests
import torch
import json
import yaml
import html
from modules import shared
from modules.models import reload_model, unload_model
from PIL import Image

torch._C._jit_set_profiling_mode(False)

# parameters which can be customized in settings.json of webui
params = {
    'address': 'http://127.0.0.1:7860',
    'mode': 0,  # modes of operation: 0 (Manual only), 1 (Immersive/Interactive - looks for words to trigger), 2 (Picturebook Adventure - Always on)
    'manage_VRAM': False,
    'save_img': False,
    'SD_model': 'NeverEndingDream',  # not used right now
    'prompt_prefix': '(Masterpiece:1.1), detailed, intricate, colorful',
    'negative_prompt': '(worst quality, low quality:1.3)',
    'width': 512,
    'height': 512,
    'denoising_strength': 0.61,
    'restore_faces': False,
    'enable_hr': False,
    'hr_upscaler': 'ESRGAN_4x',
    'hr_scale': '1.0',
    'seed': -1,
    'sampler_name': 'DDIM',
    'steps': 32,
    'cfg_scale': 7,
    'secondary_prompt': False,
    'translations': False,
    'checkpoint_prompt' : False,
    'processing': False,
    'disable_loras': False,
    'description_weight' : '1',
    'subject_weight' : '0',
    'initial_weight' : '0',
    'secondary_negative_prompt' : '',
    'secondary_positive_prompt' : '',
    'showDescription': True
}


def give_VRAM_priority(actor):
    global shared, params

    if actor == 'SD':
        unload_model()
        print("Requesting Auto1111 to re-load last checkpoint used...")
        response = requests.post(url=f'{params["address"]}/sdapi/v1/reload-checkpoint', json='')
        response.raise_for_status()

    elif actor == 'LLM':
        print("Requesting Auto1111 to vacate VRAM...")
        response = requests.post(url=f'{params["address"]}/sdapi/v1/unload-checkpoint', json='')
        response.raise_for_status()
        reload_model()

    elif actor == 'set':
        print("VRAM mangement activated -- requesting Auto1111 to vacate VRAM...")
        response = requests.post(url=f'{params["address"]}/sdapi/v1/unload-checkpoint', json='')
        response.raise_for_status()

    elif actor == 'reset':
        print("VRAM mangement deactivated -- requesting Auto1111 to reload checkpoint")
        response = requests.post(url=f'{params["address"]}/sdapi/v1/reload-checkpoint', json='')
        response.raise_for_status()

    else:
        raise RuntimeError(f'Managing VRAM: "{actor}" is not a known state!')

    response.raise_for_status()
    del response


if params['manage_VRAM']:
    give_VRAM_priority('set')
characterfocus = ""
positive_suffix = ""
negative_suffix = ""
a1111Status = {
    'sd_checkpoint' : '',
    'checkpoint_positive_prompt' : '',
    'checkpoint_negative_prompt' : ''
    }
checkpoint_list = []
samplers = ['DDIM', 'DPM++ 2M Karras']  # TODO: get the availible samplers with http://{address}}/sdapi/v1/samplers
SD_models = ['NeverEndingDream']  # TODO: get with http://{address}}/sdapi/v1/sd-models and allow user to select
initial_string = ""

picture_response = False  # specifies if the next model response should appear as a picture


def add_translations(description,triggered_array,tpatterns):
    global positive_suffix, negative_suffix
    i = 0
    for word_pair in tpatterns['pairs']:
        if triggered_array[i] != 1:
            if any(target in description for target in word_pair['descriptive_word']):
                positive_suffix = positive_suffix + ", " + word_pair['SD_positive_translation']
                negative_suffix = negative_suffix + ", " + word_pair['SD_negative_translation']
                triggered_array[i] = 1
        i = i + 1
    return triggered_array

def state_modifier(state):
    if picture_response:
        state['stream'] = False
    return state

def remove_surrounded_chars(string):
    # this expression matches to 'as few symbols as possible (0 upwards) between any asterisks' OR
    # 'as few symbols as possible (0 upwards) between an asterisk and the end of the string'
    return re.sub('\*[^\*]*?(\*|$)', '', string)


def triggers_are_in(string):
    string = remove_surrounded_chars(string)
    # regex searches for send|main|message|me (at the end of the word) followed by
    # a whole word of image|pic|picture|photo|snap|snapshot|selfie|meme(s),
    # (?aims) are regex parser flags
    return bool(re.search('(?aims)(send|mail|message|me)\\b.+?\\b(image|pic(ture)?|photo|polaroid|snap(shot)?|selfie|meme)s?\\b', string))

def request_generation(case,string):
    global characterfocus, subject
    subject = ""
    if case == 1:
        toggle_generation(True)
        characterfocus = True
        string = string.replace("yourself","you")
        after_you = string.split("you", 1)[1] # subdivide the string once by the first 'you' instance and get what's coming after it
        if after_you != '':
            string = "Describe in vivid detail as if you were describing to a blind person your current clothing and the environment. Describe in vivid detail as if you were describing to a blind person yourself performing the following action: " + after_you.strip()
            subject = after_you.strip()
        else:
            string = "Describe in vivid detail as if you were describing to a blind person your current clothing and the environment. Describe yourself in vivid detail as if you were describing to a blind person."
    elif case == 2:
        toggle_generation(True)
        subject = string.split('of', 1)[1]  # subdivide the string once by the first 'of' instance and get what's coming after it
        string = "Describe in vivid detail as if you were describing to a blind person the following: " + subject.strip()
    elif case == 3:
        toggle_generation(True)
        characterfocus = True
        string = "Describe in vivid detail as if you were describing to a blind person your appearance, your current state of clothing, your surroundings and what you are doing right now."
    return string

def string_evaluation(string):
    global characterfocus
    orig_string = string
    input_type = 0
    subjects = ['yourself', 'you']
    characterfocus = False
    if triggers_are_in(string):  # check for trigger words for generation
        string = string.lower()
        if "of" in string:
            if any(target in string for target in subjects): # the focus of the image should be on the sending character
                input_type = 1
            else:
                input_type = 2
        else:
            input_type = 3
    return request_generation(input_type,string)

def input_modifier(string):
    global characterfocus
    """
    This function is applied to your text inputs before
    they are fed into the model.
    """

    global params, initial_string
    initial_string = string
    if params['mode'] == 1:  # For immersive/interactive mode, send to string evaluation
            return string_evaluation(string)
    if params['mode'] == 2:
        characterfocus = False
        string = string.lower()
        return string
    if params['mode'] == 0:
        return string

def create_suffix():
    global params, positive_suffix, negative_suffix, characterfocus
    positive_suffix = ""
    negative_suffix = ""

    # load character data from json, yaml, or yml file
    if character != 'None':
        found_file = False
        folder1 = 'characters'
        folder2 = 'characters/instruction-following'
        for folder in [folder1, folder2]:
            for extension in ["yml", "yaml", "json"]:
                filepath = Path(f'{folder}/{character}.{extension}')
                if filepath.exists():
                    found_file = True
                    break
            if found_file:
                break
        file_contents = open(filepath, 'r', encoding='utf-8').read()
        data = json.loads(file_contents) if extension == "json" else yaml.safe_load(file_contents)

    if params['secondary_prompt']:
        positive_suffix = params['secondary_positive_prompt']
        negative_suffix = params['secondary_negative_prompt']
    if params['checkpoint_prompt']:
        if params['secondary_prompt']:
            positive_suffix = positive_suffix + ", " + a1111Status['checkpoint_positive_prompt']
            negative_suffix = negative_suffix + ", " + a1111Status['checkpoint_negative_prompt']
        else:
            positive_suffix = a1111Status['checkpoint_positive_prompt']
            negative_suffix = a1111Status['checkpoint_negative_prompt']
    if characterfocus and character != 'None':
        positive_suffix = data['sd_tags_positive'] if 'sd_tags_positive' in data else ""
        negative_suffix = data['sd_tags_negative'] if 'sd_tags_negative' in data else ""
        if params['secondary_prompt']:
            positive_suffix = params['secondary_positive_prompt'] + ", " + data['sd_tags_positive'] if 'sd_tags_positive' in data else params['secondary_positive_prompt']
            negative_suffix = params['secondary_negative_prompt'] + ", " + data['sd_tags_negative'] if 'sd_tags_negative' in data else params['secondary_negative_prompt']
        if params['checkpoint_prompt']:
            positive_suffix = positive_suffix + ", " + a1111Status['checkpoint_positive_prompt'] if 'checkpoint_positive_prompt' in a1111Status else positive_suffix
            negative_suffix = negative_suffix + ", " + a1111Status['checkpoint_negative_prompt'] if 'checkpoint_negative_prompt' in a1111Status else negative_suffix

def clean_spaces(text): # Cleanup double spaces, double commas, and comma-space-comma as these are all meaningless to us and interfere with splitting up tags
    while any([", ," in text, ",," in text, "  " in text]):
        text = text.replace(", ,", ",")
        text = text.replace(",,", ",")
        text = text.replace("  ", " ")
    try:
        while any([text[0] == " ",text[0] == ","]): # Cleanup leading spaces and commas, trailing spaces and commas
            if text[0] == " ":
                text = text.replace(" ","",1)
            if text[0] == ",":
                text = text.replace(",","",1)
        while any([text[len(text)-1] == " ",text[len(text)-1] == ","]):
            if text[len(text)-1] == " ":
                text = text[::-1].replace(" ","",1)[::-1]
            if text[len(text)-1] == ",":
                text = text[::-1].replace(",","",1)[::-1]
    except IndexError: # IndexError is expected if string is empty or becomes empty during cleanup and can be safely ignored
        pass
    except:
        print("Error cleaning up text")
    return text

def tag_calculator(affix):
    string_tags = affix
    affix = affix.replace(', ', ',')
    affix = affix.replace(' ,', ',')
    tags = affix.split(",")

    if params['processing'] == False: # A simple processor that removes exact duplicates (does not remove duplicates with different weights)
        string_tags = ""
        unique = []
        for tag in tags:
            if tag not in unique:
                    unique.append(tag)
        for tag in unique:
                string_tags += ", " + tag

    if params['processing'] == True: # A smarter processor that calculates resulting tags from multiple tags
        string_tags = ""

        class tag_objects: # Tags have three characteristics, their text, their type and their weight. The type distinguishes between simple tags without parenthesis, LORAs and weighted tags
            def __init__(self, text, tag_type, weight):
                self.text = text
                self.tag_type = tag_type
                self.weight = float(weight)

        initial_tags = []

        for tag in tags: # Create an array of all tags as objects. Use the first character in the tag to distinguish the type
            if tag:
                if tag[0] != "(" and tag[0] != "<":
                    initial_tags.append(tag_objects(tag,"simple",1.0)) # Simple tags start with neither a ( or a < and are assigned a weight of one
                if tag[0] == "<":
                    pattern = r'.*?\:(.*):(.*)\>.*'
                    match = re.search(pattern,tag)
                    initial_tags.append(tag_objects(match.group(1),"lora",match.group(2))) # LORAs start with a < and have their own weight indicated with them
                if tag[0] == "(":
                    if ":" in tag:
                        pattern = r'\((.*)\:(.*)\).*'
                        match = re.search(pattern,tag)
                        initial_tags.append(tag_objects(match.group(1),"weighted",match.group(2))) # Weighted tags start with a ( and their weight can be indicated after a :
                    else:
                        pattern = r'\((.*)\).*'
                        match = re.search(pattern,tag)
                        initial_tags.append(tag_objects(match.group(1),"weighted",1.2)) # Weighted tags sometimes don't have a weight indicated, in these cases I have assigned them an arbitrary weight of 1.2

        unique = []

        for tag in initial_tags: # Remove duplicate simple tags without parenthesis, increase weight according to repetition, convert them to weighted tags and put them back into the array so they can later be processed again as weighted tags
            if tag.tag_type == "simple":
                if any(x.text == tag.text for x in unique):
                    for matched_tag in unique:
                        if matched_tag.text == tag.text:
                            resulting_weight = matched_tag.weight + 0.1
                            matched_tag.weight = float(resulting_weight)
                else:
                    unique.append(tag_objects(tag.text,"weighted",tag.weight))
        initial_tags = initial_tags + unique

        loras = []

        for tag in initial_tags: # Remove duplicate LORAs, keep only highest weight found and put them into a separate array
            if tag.tag_type == "lora":
                if any(x.text == tag.text for x in loras):
                    for matched_tag in loras:
                        if matched_tag.text == tag.text:
                            if tag.weight > matched_tag.weight:
                                matched_tag.weight = float(tag.weight)
                else:
                    loras.append(tag_objects(tag.text,"lora",tag.weight))

        final_tags = []

        for tag in initial_tags: # Remove duplicate weighted tags and calculate final tag weight (including converted simple tags) and the unique ones with their final weight in a separate array
            if tag.tag_type == "weighted":
                if any(x.text == tag.text for x in final_tags):
                    for matched_tag in final_tags:
                        if matched_tag.text == tag.text:
                            if tag.weight == 1.0:
                                resulting_weight = matched_tag.weight + 0.1
                            else:
                                resulting_weight = matched_tag.weight + (tag.weight - 1)
                            matched_tag.weight = float(resulting_weight)
                else:
                    final_tags.append(tag_objects(tag.text,tag.tag_type,tag.weight))

        for tag in final_tags: # Construct a string from the finalized unique weighted tags and the unique LORAs to pass to the payload
            if tag.weight == 1.0:
                string_tags += tag.text + ", "
            else:
                if tag.weight > 0:
                    string_tags += "(" + tag.text + ":" + str(round(tag.weight,1)) + "), "

        if not params['disable_loras']:
            for tag in loras:
                string_tags += "<lora:" + tag.text + ":" + str(round(tag.weight,1)) + ">, "

    return string_tags

def build_body(description,subject,original):
    response = ""
    if all([description, float(params['description_weight']) != 0]):
        if float(params['description_weight']) == 1:
            response = description + ", "
        else:
            response = "(" + description + ":" + str(params['description_weight']) + "), "
    if all([subject, float(params['subject_weight']) != 0]):
        if float(params['subject_weight']) == 1:
            response += subject + ", "
        else:
            response += "(" + subject + ":" + str(params['subject_weight']) + "), "
    if all([original, float(params['initial_weight']) != 0]):
        if float(params['initial_weight']) == 1:
            response += original + ", "
        else:
            response += "(" + original + ":" + str(params['initial_weight']) + "), "
    return response

# Get and save the Stable Diffusion-generated picture
def get_SD_pictures(description):

    global subject, params, initial_string
    
    if subject is None:
        subject = ''
    
    if params['manage_VRAM']:
        give_VRAM_priority('SD')

    create_suffix()
    if params['translations']:
        tpatterns = json.loads(open(Path(f'extensions/sd_api_pictures_tag_injection/translations.json'), 'r', encoding='utf-8').read())
        if character != 'None':
            found_file = False
            folder1 = 'characters'
            folder2 = 'characters/instruction-following'
            for folder in [folder1, folder2]:
                for extension in ["yml", "yaml", "json"]:
                    filepath = Path(f'{folder}/{character}.{extension}')
                    if filepath.exists():
                        found_file = True
                        break
                if found_file:
                    break
            file_contents = open(filepath, 'r', encoding='utf-8').read()
            data = json.loads(file_contents) if extension == "json" else yaml.safe_load(file_contents)
            tpatterns['pairs'] = tpatterns['pairs'] + data['translation_patterns'] if 'translation_patterns' in data else tpatterns['pairs']
        triggered_array = [0] * len(tpatterns['pairs'])
        triggered_array = add_translations(initial_string,triggered_array,tpatterns)
        add_translations(description,triggered_array,tpatterns)

    final_positive_prompt = html.unescape(clean_spaces(tag_calculator(clean_spaces(params['prompt_prefix'])) + ", " + build_body(description,subject,initial_string) + tag_calculator(clean_spaces(positive_suffix))))
    final_negative_prompt = html.unescape(clean_spaces(tag_calculator(clean_spaces(params['negative_prompt'])) + ", " + tag_calculator(clean_spaces(negative_suffix))))

    payload = {
        "prompt": final_positive_prompt,
        "negative_prompt": final_negative_prompt,
        "seed": params['seed'],
        "sampler_name": params['sampler_name'],
        "enable_hr": params['enable_hr'],
        "hr_scale": params['hr_scale'],
        "hr_upscaler": params['hr_upscaler'],
        "denoising_strength": params['denoising_strength'],
        "steps": params['steps'],
        "cfg_scale": params['cfg_scale'],
        "width": params['width'],
        "height": params['height'],
        "restore_faces": params['restore_faces'],
        "override_settings_restore_afterwards": True
    }

    print(f'Prompting the image generator via the API on {params["address"]}...')
    response = requests.post(url=f'{params["address"]}/sdapi/v1/txt2img', json=payload)
    response.raise_for_status()
    r = response.json()

    visible_result = ""
    for img_str in r['images']:
        if params['save_img']:
            img_data = base64.b64decode(img_str)

            variadic = f'{date.today().strftime("%Y_%m_%d")}/{character}_{int(time.time())}'
            output_file = Path(f'extensions/sd_api_pictures_tag_injection/outputs/{variadic}.png')
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file.as_posix(), 'wb') as f:
                f.write(img_data)

            visible_result = visible_result + f'<img src="/file/extensions/sd_api_pictures_tag_injection/outputs/{variadic}.png" alt="{description}" style="max-width: unset; max-height: unset;">\n'
        else:
            image = Image.open(io.BytesIO(base64.b64decode(img_str.split(",", 1)[0])))
            # lower the resolution of received images for the chat, otherwise the log size gets out of control quickly with all the base64 values in visible history
            image.thumbnail((300, 300))
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            buffered.seek(0)
            image_bytes = buffered.getvalue()
            img_str = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode()
            visible_result = visible_result + f'<img src="{img_str}" alt="{description}">\n'
            
    if params['manage_VRAM']:
        give_VRAM_priority('LLM')

    return visible_result

# TODO: how do I make the UI history ignore the resulting pictures (I don't want HTML to appear in history)
# and replace it with 'text' for the purposes of logging?
def output_modifier(string, state):
    """
    This function is applied to the model outputs.
    """

    global picture_response, params, character
    
    character = state.get('character_menu','None')

    if not picture_response:
        return string

    string = remove_surrounded_chars(string)
    string = string.replace('"', '')
    string = string.replace('“', '')
    string = string.replace('\n', ' ')
    string = string.strip()

    if string == '':
        string = 'no viable description in reply, try regenerating'
        return string

    text = ""
    if (params['mode'] < 2):
        toggle_generation(False)
        text = f'*Sends a picture which portrays: “{string}”*'
    else:
        text = string

    string = get_SD_pictures(string)
    
    if params['showDescription']:
        string = string + "\n" + text

    return string


def bot_prefix_modifier(string):
    """
    This function is only applied in chat mode. It modifies
    the prefix text for the Bot and can be used to bias its
    behavior.
    """

    return string


def toggle_generation(*args):
    global picture_response, shared

    if not args:
        picture_response = not picture_response
    else:
        picture_response = args[0]

    shared.processing_message = "*Is sending a picture...*" if picture_response else "*Is typing...*"


def filter_address(address):
    address = address.strip()
    # address = re.sub('http(s)?:\/\/|\/$','',address) # remove starting http:// OR https:// OR trailing slash
    address = re.sub('\/$', '', address)  # remove trailing /s
    if not address.startswith('http'):
        address = 'http://' + address
    return address


def SD_api_address_update(address):

    global params

    msg = "✔️ SD API is found on:"
    address = filter_address(address)
    params.update({"address": address})
    try:
        response = requests.get(url=f'{params["address"]}/sdapi/v1/sd-models')
        response.raise_for_status()
        # r = response.json()
    except:
        msg = "❌ No SD API endpoint on:"

    return gr.Textbox.update(label=msg)

def get_checkpoints():
    global a1111Status, checkpoint_list

    models = requests.get(url=f'{params["address"]}/sdapi/v1/sd-models')
    options = requests.get(url=f'{params["address"]}/sdapi/v1/options')
    options_json = options.json()
    a1111Status['sd_checkpoint'] = options_json['sd_model_checkpoint']
    checkpoint_list = [result["title"] for result in models.json()]
    return gr.update(choices=checkpoint_list, value=a1111Status['sd_checkpoint'])

def load_checkpoint(checkpoint):
    global a1111Status
    a1111Status['checkpoint_positive_prompt'] = ""
    a1111Status['checkpoint_negative_prompt'] = ""

    payload = {
        "sd_model_checkpoint": checkpoint
    }

    prompts = json.loads(open(Path(f'extensions/sd_api_pictures_tag_injection/checkpoints.json'), 'r', encoding='utf-8').read())
    for pair in prompts['pairs']:
        if pair['name'] == a1111Status['sd_checkpoint']:
            a1111Status['checkpoint_positive_prompt'] = pair['positive_prompt']
            a1111Status['checkpoint_negative_prompt'] = pair['negative_prompt']
    requests.post(url=f'{params["address"]}/sdapi/v1/options', json=payload)

def get_samplers():
    global params
    
    try:
        response = requests.get(url=f'{params["address"]}/sdapi/v1/samplers')
        response.raise_for_status()
        samplers = [x["name"] for x in response.json()]
    except:
        samplers = []

    return gr.update(choices=samplers)

def ui():

    # Gradio elements
    # gr.Markdown('### Stable Diffusion API Pictures') # Currently the name of extension is shown as the title
    with gr.Accordion("Parameters", open=True):
        with gr.Row():
            address = gr.Textbox(placeholder=params['address'], value=params['address'], label='Auto1111\'s WebUI address')
            modes_list = ["Manual", "Immersive/Interactive", "Picturebook/Adventure"]
            mode = gr.Dropdown(modes_list, value=modes_list[params['mode']], allow_custom_value=True, label="Mode of operation", type="index")
            with gr.Column(scale=1, min_width=300):
                manage_VRAM = gr.Checkbox(value=params['manage_VRAM'], label='Manage VRAM')
                save_img = gr.Checkbox(value=params['save_img'], label='Keep original images and use them in chat')
                secondary_prompt = gr.Checkbox(value=params['secondary_prompt'], label='Add secondary tags in prompt')
                translations = gr.Checkbox(value=params['translations'], label='Activate SD translations')
                tag_processing = gr.Checkbox(value=params['processing'], label='Advanced tag processing')
                disable_loras = gr.Checkbox(value=params['disable_loras'], label='Disable SD LORAs')
            force_pic = gr.Button("Force the picture response")
            suppr_pic = gr.Button("Suppress the picture response")
        with gr.Row():
            checkpoint = gr.Dropdown(checkpoint_list, value=a1111Status['sd_checkpoint'], allow_custom_value=True, label="Checkpoint", type="value")
            checkpoint_prompt = gr.Checkbox(value=params['checkpoint_prompt'], label='Add checkpoint tags in prompt')
            update_checkpoints = gr.Button("Get list of checkpoints")

        with gr.Accordion("Description mixer", open=False):
            description_weight = gr.Slider(0, 4, value=params['description_weight'], step=0.1, label='LLM Response Weight')
            subject_weight = gr.Slider(0, 4, value=params['subject_weight'], step=0.1, label='Subject Weight')
            initial_weight = gr.Slider(0, 4, value=params['initial_weight'], step=0.1, label='Initial Prompt Weight')

        with gr.Accordion("Generation parameters", open=False):
            prompt_prefix = gr.Textbox(placeholder=params['prompt_prefix'], value=params['prompt_prefix'], label='Prompt Prefix (best used to describe the look of the character)')
            negative_prompt = gr.Textbox(placeholder=params['negative_prompt'], value=params['negative_prompt'], label='Negative Prompt')
            with gr.Row():
                with gr.Column():
                    secondary_positive_prompt = gr.Textbox(placeholder=params['secondary_positive_prompt'], value=params['secondary_positive_prompt'], label='Secondary positive prompt')
                with gr.Column():
                    secondary_negative_prompt = gr.Textbox(placeholder=params['secondary_negative_prompt'], value=params['secondary_negative_prompt'], label='Secondary negative prompt')
            with gr.Row():
                with gr.Column():
                    width = gr.Slider(64, 2048, value=params['width'], step=64, label='Width')
                    height = gr.Slider(64, 2048, value=params['height'], step=64, label='Height')
                with gr.Column():
                    with gr.Row():
                        sampler_name = gr.Dropdown(value=params['sampler_name'],allow_custom_value=True,label='Sampling method', elem_id="sampler_box")
                        update_samplers = gr.Button("Get samplers")
                    steps = gr.Slider(1, 150, value=params['steps'], step=1, label="Sampling steps")
            with gr.Row():
                seed = gr.Number(label="Seed", value=params['seed'], elem_id="seed_box")
                cfg_scale = gr.Number(label="CFG Scale", value=params['cfg_scale'], elem_id="cfg_box")
                with gr.Column() as hr_options:
                    restore_faces = gr.Checkbox(value=params['restore_faces'], label='Restore faces')
                    enable_hr = gr.Checkbox(value=params['enable_hr'], label='Hires. fix')
            with gr.Row(visible=params['enable_hr'], elem_classes="hires_opts") as hr_options:
                    hr_scale = gr.Slider(1, 4, value=params['hr_scale'], step=0.1, label='Upscale by')
                    denoising_strength = gr.Slider(0, 1, value=params['denoising_strength'], step=0.01, label='Denoising strength')
                    hr_upscaler = gr.Textbox(placeholder=params['hr_upscaler'], value=params['hr_upscaler'], label='Upscaler')

    # Event functions to update the parameters in the backend
    address.change(lambda x: params.update({"address": filter_address(x)}), address, None)
    mode.select(lambda x: params.update({"mode": x}), mode, None)
    mode.select(lambda x: toggle_generation(x > 1), inputs=mode, outputs=None)
    manage_VRAM.change(lambda x: params.update({"manage_VRAM": x}), manage_VRAM, None)
    manage_VRAM.change(lambda x: give_VRAM_priority('set' if x else 'reset'), inputs=manage_VRAM, outputs=None)
    save_img.change(lambda x: params.update({"save_img": x}), save_img, None)

    address.submit(fn=SD_api_address_update, inputs=address, outputs=address)
    description_weight.change(lambda x: params.update({"description_weight": x}), description_weight, None)
    initial_weight.change(lambda x: params.update({"initial_weight": x}), initial_weight, None)
    subject_weight.change(lambda x: params.update({"subject_weight": x}), subject_weight, None)
    prompt_prefix.change(lambda x: params.update({"prompt_prefix": x}), prompt_prefix, None)
    negative_prompt.change(lambda x: params.update({"negative_prompt": x}), negative_prompt, None)
    width.change(lambda x: params.update({"width": x}), width, None)
    height.change(lambda x: params.update({"height": x}), height, None)
    hr_scale.change(lambda x: params.update({"hr_scale": x}), hr_scale, None)
    denoising_strength.change(lambda x: params.update({"denoising_strength": x}), denoising_strength, None)
    restore_faces.change(lambda x: params.update({"restore_faces": x}), restore_faces, None)
    hr_upscaler.change(lambda x: params.update({"hr_upscaler": x}), hr_upscaler, None)
    enable_hr.change(lambda x: params.update({"enable_hr": x}), enable_hr, None)
    enable_hr.change(lambda x: hr_options.update(visible=params["enable_hr"]), enable_hr, hr_options)
    tag_processing.change(lambda x: params.update({"processing": x}), tag_processing, None)
    tag_processing.change(lambda x: disable_loras.update(visible=params["processing"]), tag_processing, disable_loras)
    disable_loras.change(lambda x: params.update({"disable_loras": x}), disable_loras, None)

    update_checkpoints.click(get_checkpoints, None, checkpoint)
    checkpoint.change(lambda x: a1111Status.update({"sd_checkpoint": x}), checkpoint, None)
    checkpoint.change(load_checkpoint, checkpoint, None)
    checkpoint_prompt.change(lambda x: params.update({"checkpoint_prompt": x}), checkpoint_prompt, None)

    translations.change(lambda x: params.update({"translations": x}), translations, None)
    secondary_prompt.change(lambda x: params.update({"secondary_prompt": x}), secondary_prompt, None)
    secondary_positive_prompt.change(lambda x: params.update({"secondary_positive_prompt": x}), secondary_positive_prompt, None)
    secondary_negative_prompt.change(lambda x: params.update({"secondary_negative_prompt": x}), secondary_negative_prompt, None)

    update_samplers.click(get_samplers, None, sampler_name)
    sampler_name.change(lambda x: params.update({"sampler_name": x}), sampler_name, None)
    steps.change(lambda x: params.update({"steps": x}), steps, None)
    seed.change(lambda x: params.update({"seed": x}), seed, None)
    cfg_scale.change(lambda x: params.update({"cfg_scale": x}), cfg_scale, None)

    force_pic.click(lambda x: toggle_generation(True), inputs=force_pic, outputs=None)
    suppr_pic.click(lambda x: toggle_generation(False), inputs=suppr_pic, outputs=None)
