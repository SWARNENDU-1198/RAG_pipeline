"""
stt/sarvam_client.py

High-Resilience Speech-to-Text (STT) Client:
- Primary STT: Sarvam AI (saarika:v2.5 / saaras:v3) for fast Indic speech-to-text.
- Multimodal Fallback STT: Google Gemini Multimodal Audio (gemini-3.6-flash) for 100% cloud uptime.
- Text-to-Speech (TTS): Sarvam AI (bulbul:v2).
"""

import os
import time
import base64
import logging
import httpx
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"
SARVAM_TRANSLATE_URL = "https://api.sarvam.ai/translate"


class SarvamSTTClient:
    """Client for Sarvam AI Speech-to-Text & Text-to-Speech REST APIs with Gemini Multimodal fallback."""

    def __init__(self, api_key: str = None, max_retries: int = 2, backoff_factor: float = 1.5):
        self.api_key = api_key or os.getenv("SARVAM_API_KEY", "")
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self._client = httpx.Client(
            timeout=15.0,
            limits=httpx.Limits(max_keepalive_connections=15, max_connections=30)
        )

    def _transcribe_gemini_multimodal(self, audio_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Transcribes audio using Google Gemini Multimodal Audio understanding."""
        gemini_key = os.getenv("GEMINI_API_KEY", "")
        if not gemini_key or not audio_bytes or len(audio_bytes) < 100:
            return {"transcript": "", "language_code": "unknown", "confidence": 0.0, "provider": "none"}

        mime = "audio/wav"
        if filename.endswith(".webm") or (len(audio_bytes) > 4 and audio_bytes[:4] == b"\x1a\x45\xdf\xa3"):
            mime = "audio/webm"
        elif filename.endswith(".mp3"):
            mime = "audio/mp3"
        elif filename.endswith(".ogg"):
            mime = "audio/ogg"

        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        payload = {
            "contents": [
                {
                    "parts": [
                        {"inlineData": {"mimeType": mime, "data": audio_b64}},
                        {"text": "Transcribe the spoken audio verbatim in its original language and script (Hindi, Tamil, Telugu, Bengali, Kannada, Malayalam, Marathi, Gujarati, Punjabi, Odia, Assamese, Urdu, English, or Hinglish). Output ONLY the transcript without quotes or explanations. If the audio is silent or only background tone/noise, reply with NO_SPEECH."}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 250
            }
        }

        candidate_models = ["gemini-3.6-flash", "gemini-3.7-flash", "gemini-flash-latest", "gemini-2.5-flash"]
        for m in candidate_models:
            endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={gemini_key}"
            try:
                res = self._client.post(endpoint, json=payload, headers={"Content-Type": "application/json"}, timeout=15.0)
                if res.status_code == 200:
                    data = res.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts and "text" in parts[0]:
                            text = parts[0]["text"].strip()
                            if text and "NO_SPEECH" not in text:
                                logger.info(f"Gemini Multimodal STT success ({m}): transcript='{text}'")
                                return {
                                    "transcript": text,
                                    "language_code": "auto",
                                    "confidence": 0.95,
                                    "provider": f"gemini_multimodal ({m})"
                                }
            except Exception as e:
                logger.warning(f"Gemini audio transcription attempt on {m} failed: {e}")

        return {"transcript": "", "language_code": "unknown", "confidence": 0.0, "provider": "gemini_multimodal_failed"}

    def transcribe_audio(
        self,
        audio_bytes: bytes,
        language_code: str = "unknown",
        filename: str = "audio.wav",
        model: str = "saarika:v2.5"
    ) -> Dict[str, Any]:
        """
        Transcribes audio bytes to text using Sarvam AI STT API with Gemini Multimodal fallback.
        """
        if not audio_bytes or len(audio_bytes) < 100:
            logger.warning("Empty or truncated audio bytes provided to STT.")
            return {
                "transcript": "",
                "language_code": "unknown",
                "confidence": 0.0,
                "provider": "none",
                "error": "Empty audio data"
            }

        # 1. Try Sarvam AI STT if API key is present
        if self.api_key:
            headers = {
                "api-subscription-key": self.api_key
            }

            content_type = "audio/wav"
            if filename.endswith(".webm") or (len(audio_bytes) > 4 and audio_bytes[:4] == b"\x1a\x45\xdf\xa3"):
                content_type = "audio/webm"
                if not filename.endswith(".webm"):
                    filename = "audio.webm"
            elif filename.endswith(".mp3"):
                content_type = "audio/mp3"

            files = {
                "file": (filename, audio_bytes, content_type)
            }
            
            candidate_models = [model, "saaras:v3", "saarika:v2.5"]
            candidate_models = list(dict.fromkeys(candidate_models))

            for m in candidate_models:
                data = {
                    "model": m,
                    "language_code": language_code if language_code else "unknown"
                }

                for attempt in range(self.max_retries + 1):
                    try:
                        logger.info(f"Sending STT request to Sarvam AI [model: {m}, lang: {data['language_code']}] (attempt {attempt + 1})...")
                        response = self._client.post(SARVAM_STT_URL, headers=headers, data=data, files=files)
                            
                        if response.status_code == 200:
                            res_json = response.json()
                            transcript = res_json.get("transcript", res_json.get("text", "")).strip()
                            detected_lang = res_json.get("language_code", language_code)
                            conf = res_json.get("language_probability", res_json.get("confidence", 0.95))
                            
                            if transcript:
                                logger.info(f"Sarvam STT success: transcript='{transcript}', lang='{detected_lang}'")
                                return {
                                    "transcript": transcript,
                                    "language_code": detected_lang,
                                    "confidence": float(conf) if conf is not None else 0.95,
                                    "provider": f"sarvam_ai ({m})",
                                    "raw_response": res_json
                                }
                        else:
                            logger.warning(f"Sarvam STT ({m}) returned {response.status_code}: {response.text}")
                            if response.status_code == 400 and "model" in response.text.lower():
                                break
                    except Exception as e:
                        logger.warning(f"Sarvam STT request failed on attempt {attempt + 1}: {e}")

                    if attempt < self.max_retries:
                        time.sleep(self.backoff_factor ** attempt)

        # 2. Fallback to Gemini Multimodal Audio Transcription
        logger.info("Falling back to Google Gemini Multimodal Audio Transcription...")
        gemini_res = self._transcribe_gemini_multimodal(audio_bytes, filename)
        if gemini_res.get("transcript"):
            return gemini_res

        return {
            "transcript": "",
            "language_code": "unknown",
            "confidence": 0.0,
            "provider": "stt_exhausted",
            "error": "Could not transcribe audio"
        }

    def synthesize_speech(
        self,
        text: str,
        target_language_code: str = "hi-IN",
        speaker: str = "anushka"
    ) -> Optional[str]:
        """
        Synthesizes text to speech using Sarvam AI TTS API.
        Returns base64 encoded audio string (WAV format) or None on failure.
        """
        if not self.api_key or not text or not text.strip():
            return None

        headers = {
            "api-subscription-key": self.api_key,
            "Content-Type": "application/json"
        }

        valid_tts_langs = {
            "hi": "hi-IN", "hi-in": "hi-IN",
            "bn": "bn-IN", "bn-in": "bn-IN",
            "ta": "ta-IN", "ta-in": "ta-IN",
            "te": "te-IN", "te-in": "te-IN",
            "kn": "kn-IN", "kn-in": "kn-IN",
            "ml": "ml-IN", "ml-in": "ml-IN",
            "mr": "mr-IN", "mr-in": "mr-IN",
            "gu": "gu-IN", "gu-in": "gu-IN",
            "pa": "pa-IN", "pa-in": "pa-IN",
            "od": "od-IN", "od-in": "od-IN",
            "en": "en-IN", "en-in": "en-IN"
        }
        tts_lang = valid_tts_langs.get(target_language_code.lower(), "hi-IN")

        payload = {
            "inputs": [text[:500]],
            "target_language_code": tts_lang,
            "speaker": speaker,
            "model": "bulbul:v2"
        }

        try:
            res = self._client.post(SARVAM_TTS_URL, headers=headers, json=payload)
            if res.status_code == 200:
                data = res.json()
                audios = data.get("audios", [])
                if audios and len(audios) > 0:
                    return audios[0]
            else:
                logger.warning(f"Sarvam TTS failed with {res.status_code}: {res.text}")
        except Exception as e:
            logger.warning(f"Sarvam TTS exception: {e}")

        return None

    def translate_text(
        self,
        text: str,
        source_language_code: str = "en-IN",
        target_language_code: str = "hi-IN"
    ) -> Optional[str]:
        """Translates text using Sarvam AI translation API."""
        if not self.api_key or not text or not text.strip():
            return None

        headers = {
            "api-subscription-key": self.api_key,
            "Content-Type": "application/json"
        }

        payload = {
            "input": text,
            "source_language_code": source_language_code,
            "target_language_code": target_language_code,
            "speaker_gender": "Female",
            "mode": "formal"
        }

        try:
            res = self._client.post(SARVAM_TRANSLATE_URL, headers=headers, json=payload)
            if res.status_code == 200:
                data = res.json()
                return data.get("translated_text", text)
        except Exception as e:
            logger.warning(f"Sarvam translate exception: {e}")
        return None
