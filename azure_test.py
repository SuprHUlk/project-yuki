import base64
import json
import mimetypes
import os
from dotenv import load_dotenv
from openai import OpenAI
import azure.cognitiveservices.speech as speechsdk

load_dotenv()

# =========================================================
# CONFIG
# =========================================================

API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

# Azure AI Foundry / OpenAI endpoint
OPENAI_ENDPOINT = "https://project-yuki-4-resource.services.ai.azure.com/openai/v1"

# Azure Speech endpoint
SPEECH_ENDPOINT = "https://project-yuki-4-resource.cognitiveservices.azure.com/"

DEPLOYMENT_NAME = "o4-mini"

IMAGE_PATH = "elden_ring_player_death_composite.jpg"

VOICE = "en-US-Bree:DragonHDLatestNeural"

# =========================================================
# OPENAI CLIENT
# =========================================================

client = OpenAI(
    base_url=OPENAI_ENDPOINT,
    api_key=API_KEY,
)

# =========================================================
# HELPERS
# =========================================================

def image_to_data_url(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"

    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime};base64,{b64}"


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


# =========================================================
# SSML GENERATOR
# =========================================================

def build_ssml(data):
    emotion = data.get("emotion", "chat")
    text = data["line"]

    intensity = float(data.get("intensity", 0.5))
    intensity = clamp(intensity, 0.0, 1.0)

    # -----------------------------------------------------
    # Emotion presets
    # -----------------------------------------------------

    if emotion == "excited":
        style = "excited"

        pitch_value = int(6 + intensity * 10)
        rate_value = int(8 + intensity * 10)

        pitch = f"+{pitch_value}%"
        rate = f"+{rate_value}%"

    elif emotion == "fearful":
        style = "fearful"

        pitch_value = int(3 + intensity * 8)
        rate_value = int(4 + intensity * 14)

        pitch = f"+{pitch_value}%"
        rate = f"+{rate_value}%"

    elif emotion == "sad":
        style = "sad"

        pitch_value = int(2 + intensity * 4)
        rate_value = int(8 + intensity * 10)

        pitch = f"-{pitch_value}%"
        rate = f"-{rate_value}%"

    elif emotion == "empathetic":
        style = "empathetic"

        pitch_value = int(1 + intensity * 3)
        rate_value = int(4 + intensity * 8)

        pitch = f"-{pitch_value}%"
        rate = f"-{rate_value}%"

    else:
        style = "chat"
        pitch = "0%"
        rate = "0%"

    # -----------------------------------------------------
    # Dynamic pauses
    # -----------------------------------------------------

    processed_text = text

    processed_text = processed_text.replace("...", '<break time="250ms"/>')
    processed_text = processed_text.replace("—", '<break time="180ms"/>')
    processed_text = processed_text.replace("!", '!<break time="120ms"/>')

    # -----------------------------------------------------
    # Final SSML
    # -----------------------------------------------------

    ssml = f"""
<speak version="1.0"
xmlns="http://www.w3.org/2001/10/synthesis"
xmlns:mstts="http://www.w3.org/2001/mstts"
xml:lang="en-US">

    <voice name="{VOICE}">

        <mstts:express-as style="{style}">

            <prosody rate="{rate}" pitch="{pitch}">

                {processed_text}

            </prosody>

        </mstts:express-as>

    </voice>

</speak>
"""

    return ssml


# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """
You are Yuki, a witty anime-style in-game companion narrator.

You are shown a composite image made from consecutive gameplay frames.

You must:
1. Classify the gameplay moment
2. Generate a natural spoken reaction
3. Generate emotional metadata

==================================================
CLASSIFICATION LABELS
==================================================

Choose EXACTLY ONE:

PLAYER_DEATH
- Player died
- YOU DIED / GAME OVER / DEFEAT
- death fade
- respawn
- ragdoll
- death takes priority

BOSS_FIGHT
- Boss healthbar
- Named major enemy
- Cinematic encounter
- Major boss battle

OTHER
- Exploration
- Menus
- Traversal
- Normal combat
- Dialogue
- Inventory
- Loading

If uncertain -> OTHER

==================================================
YUKI PERSONALITY
==================================================

Yuki is:
- witty
- playful
- emotionally reactive
- supportive
- slightly sarcastic
- energetic during combat
- emotionally attached to the player

Yuki sounds NATURAL.

Use:
- hesitation
- interruptions
- short dramatic reactions
- conversational delivery

Examples:
"Wait... phase two?!"
"Nope. Absolutely not."
"You almost had that..."
"DODGE LEFT!"

==================================================
LINE RULES
==================================================

- One sentence
- Max 15 words
- Spoken English
- No markdown
- No emoji
- No stage directions
- Speak directly to player
- Avoid naming game unless visible

==================================================
EMOTION RULES
==================================================

PLAYER_DEATH:
- emotion = sad OR empathetic

BOSS_FIGHT:
- emotion = excited OR fearful

OTHER:
- emotion = chat

==================================================
OUTPUT FORMAT
==================================================

Return ONLY valid JSON:

{
  "label": "PLAYER_DEATH" | "BOSS_FIGHT" | "OTHER",
  "emotion": "excited" | "fearful" | "sad" | "empathetic" | "chat",
  "intensity": 0.0 to 1.0,
  "line": "..."
}
"""

USER_PROMPT = "Classify this gameplay clip and react as Yuki."

# =========================================================
# CALL VISION MODEL
# =========================================================

response = client.chat.completions.create(
    model=DEPLOYMENT_NAME,
    temperature=1.0,
    response_format={"type": "json_object"},
    messages=[
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": USER_PROMPT,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_to_data_url(IMAGE_PATH),
                        "detail": "high",
                    },
                },
            ],
        },
    ],
)

# =========================================================
# PARSE RESPONSE
# =========================================================

data = json.loads(response.choices[0].message.content)

print("\n============================")
print("YUKI ANALYSIS")
print("============================")

print("Label     :", data["label"])
print("Emotion   :", data["emotion"])
print("Intensity :", data["intensity"])
print("Line      :", data["line"])

# =========================================================
# BUILD SSML
# =========================================================

ssml = build_ssml(data)

print("\n============================")
print("GENERATED SSML")
print("============================")

print(ssml)

# =========================================================
# AZURE SPEECH CONFIG
# =========================================================

speech_config = speechsdk.SpeechConfig(
    subscription=API_KEY,
    endpoint=SPEECH_ENDPOINT,
)

speech_config.speech_synthesis_voice_name = VOICE

audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)

speech_synthesizer = speechsdk.SpeechSynthesizer(
    speech_config=speech_config,
    audio_config=audio_config,
)

# =========================================================
# SPEAK
# =========================================================

result = speech_synthesizer.speak_ssml_async(ssml).get()

# =========================================================
# RESULT CHECK
# =========================================================

if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
    print("\nSpeech synthesized successfully.")

elif result.reason == speechsdk.ResultReason.Canceled:
    cancellation_details = result.cancellation_details

    print("\nSpeech synthesis canceled.")
    print("Reason:", cancellation_details.reason)

    if cancellation_details.reason == speechsdk.CancellationReason.Error:
        print("Error details:", cancellation_details.error_details)
