import os
import torch
from PIL import Image, ImageEnhance, ImageFilter
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    AutoTokenizer, AutoModelForSeq2SeqLM,
    VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer as AutoTokenizer2
)
from gtts import gTTS
import pygame
import platform
import tempfile
import time
import warnings
import uuid
import numpy as np
import cv2
import base64
import io

warnings.filterwarnings('ignore')

class MultilingualImageToSpeech:
    def __init__(self, audio_output_dir="outputs/audio_clips"):
        print("Initializing Enhanced Multilingual Image-to-Speech Model...")
        os.makedirs(audio_output_dir, exist_ok=True)
        self.audio_output_dir = audio_output_dir

        pygame.mixer.init()

        # Load multiple captioning models for better accuracy
        print("Loading BLIP model...")
        self.caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        
        # Load additional captioning model for cross-validation
        try:
            print("Loading ViT-GPT2 model...")
            self.vit_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
            self.vit_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
            self.vit_tokenizer = AutoTokenizer2.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
            self.use_vit = True
        except Exception as e:
            print(f"ViT-GPT2 model not available: {e}")
            self.use_vit = False

        # Updated language codes - removed Assamese, improved voice support
        self.lang_codes = {
            'english': 'en', 
            'hindi': 'hi', 
            'bengali': 'bn',
            'telugu': 'te', 
            'tamil': 'ta', 
            'malayalam': 'ml'
        }
        
        # Updated TTS supported languages with better compatibility
        self.tts_supported = {'en', 'hi', 'bn', 'ta', 'te', 'ml'}
        self.supported_languages = list(self.lang_codes.keys())

        self.translation_models = {}
        self.translation_tokenizers = {}

        self.init_translation_models()
        self.load_nllb_models()

        print("Model initialization complete!")

    def init_translation_models(self):
        model_mappings = {
            'hindi': "Helsinki-NLP/opus-mt-en-hi",
            'malayalam': "Helsinki-NLP/opus-mt-en-ml"
        }
        for lang, model_name in model_mappings.items():
            self.load_model(lang, model_name)

    def load_model(self, lang, model_name):
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.translation_tokenizers[lang] = tokenizer
            self.translation_models[lang] = model
            print(f"✅ Loaded {lang} translation model")
        except Exception as e:
            print(f"⚠ Could not load {lang} model: {e}")
            self.translation_tokenizers[lang] = None
            self.translation_models[lang] = None

    def load_nllb_models(self):
        try:
            self.nllb_tokenizer = AutoTokenizer.from_pretrained(
                "facebook/nllb-200-distilled-600M", use_fast=False
            )
            self.nllb_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
            self.nllb_lang_codes = {
                'bengali': 'ben_Beng',
                'telugu': 'tel_Telu', 
                'tamil': 'tam_Taml'
            }
            print("✅ Loaded NLLB translation model")
        except Exception as e:
            print(f"⚠ Could not load NLLB model: {e}")

    def enhance_image_quality(self, image):
        """Enhance image quality for better captioning"""
        try:
            # Convert PIL to OpenCV format
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            lab = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # Convert back to PIL
            enhanced_pil = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
            
            # Additional PIL enhancements
            enhancer = ImageEnhance.Contrast(enhanced_pil)
            enhanced_pil = enhancer.enhance(1.2)
            
            enhancer = ImageEnhance.Sharpness(enhanced_pil)
            enhanced_pil = enhancer.enhance(1.1)
            
            return enhanced_pil
        except Exception as e:
            print(f"Image enhancement failed: {e}")
            return image

    def generate_caption_blip(self, image):
        """Generate caption using BLIP model"""
        try:
            # Try multiple prompts for better context
            prompts = [
                "a photo of",
                "this image shows",
                "in this picture there is"
            ]
            
            captions = []
            for prompt in prompts:
                inputs = self.caption_processor(image, text=prompt, return_tensors="pt")
                with torch.no_grad():
                    out = self.caption_model.generate(
                        **inputs, 
                        max_length=100, 
                        num_beams=5,
                        temperature=0.7,
                        do_sample=True
                    )
                caption = self.caption_processor.decode(out[0], skip_special_tokens=True)
                captions.append(caption)
            
            # Return the most detailed caption
            return max(captions, key=len)
        except Exception as e:
            print(f"BLIP captioning failed: {e}")
            return None

    def generate_caption_vit(self, image):
        """Generate caption using ViT-GPT2 model"""
        if not self.use_vit:
            return None
            
        try:
            pixel_values = self.vit_processor(image, return_tensors="pt").pixel_values
            
            with torch.no_grad():
                output_ids = self.vit_model.generate(
                    pixel_values, 
                    max_length=50, 
                    num_beams=4,
                    temperature=0.8,
                    do_sample=True
                )
            
            caption = self.vit_tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            print(f"ViT captioning failed: {e}")
            return None

    def generate_caption(self, image):
        """Generate caption using multiple models and select the best one"""
        try:
            # Enhance image quality first
            enhanced_image = self.enhance_image_quality(image)
            
            # Get captions from both models
            blip_caption = self.generate_caption_blip(enhanced_image)
            vit_caption = self.generate_caption_vit(enhanced_image)
            
            print(f"BLIP Caption: {blip_caption}")
            if vit_caption:
                print(f"ViT Caption: {vit_caption}")
            
            # Select the best caption (longer and more descriptive)
            captions = [cap for cap in [blip_caption, vit_caption] if cap and len(cap.strip()) > 5]
            
            if not captions:
                return "Unable to describe the image clearly."
            
            # Return the most descriptive caption
            best_caption = max(captions, key=lambda x: len(x.split()))
            
            # Clean up the caption
            if best_caption.lower().startswith(('a photo of', 'this image shows', 'in this picture there is')):
                words = best_caption.split()
                if len(words) > 3:
                    best_caption = ' '.join(words[3:])
            
            return best_caption.strip()
            
        except Exception as e:
            print(f"Error generating caption: {e}")
            return "Unable to describe the image."

    def translate_text(self, text, target_language):
        try:
            lang_key = target_language.lower()

            if (lang_key in self.translation_models and 
                self.translation_models[lang_key] is not None and
                self.translation_tokenizers[lang_key] is not None):

                tokenizer = self.translation_tokenizers[lang_key]
                model = self.translation_models[lang_key]
                inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_length=128, num_beams=4)
                translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"✅ Translated to {target_language}: {translated}")
                return translated

            elif lang_key in self.nllb_lang_codes:
                tgt_lang = self.nllb_lang_codes[lang_key]
                tokenizer = self.nllb_tokenizer
                model = self.nllb_model

                tokenizer.src_lang = "eng_Latn"
                encoded = tokenizer(text, return_tensors="pt")
                generated = model.generate(**encoded, forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang))
                translated = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
                print(f"✅ Translated to {target_language}: {translated}")
                return translated

            else:
                print(f"⚠ No translation model for {target_language}. Returning English.")
                return text

        except Exception as e:
            print(f"❌ Translation failed for {target_language}: {e}")
            return text

    def generate_tts_audio_base64(self, text, language_code):
        """Generate TTS audio and return as base64 string"""
        try:
            if language_code not in self.tts_supported:
                print(f"❌ TTS generation failed: Language not supported by gTTS: {language_code}")
                return None
                
            # Create TTS object
            tts = gTTS(text=text, lang=language_code, slow=False)
            
            # Save to a BytesIO buffer instead of file
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            
            # Convert to base64
            audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
            
            print(f"✅ Generated TTS audio for {language_code}, length: {len(audio_base64)} chars")
            return audio_base64
            
        except Exception as e:
            print(f"❌ TTS generation failed for {language_code}: {e}")
            return None

    def generate_tts_audio(self, text, language_code, filename):
        """Generate TTS audio file (legacy method for local testing)"""
        try:
            if language_code not in self.tts_supported:
                print(f"❌ TTS generation failed: Language not supported by gTTS: {language_code}")
                return None
            tts = gTTS(text=text, lang=language_code, slow=False)
            audio_path = os.path.join(self.audio_output_dir, filename)
            tts.save(audio_path)
            return audio_path
        except Exception as e:
            print(f"❌ TTS generation failed: {e}")
            return None

    def play_audio(self, audio_path):
        try:
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)
            return True
        except Exception as e:
            print(f"❌ Audio playback failed: {e}")
            return False

    def text_to_speech(self, text, language='english', save_audio=True, play_audio=True):
        lang_code = self.lang_codes.get(language.lower(), 'en')
        filename = f"{language.lower()}_{uuid.uuid4().hex[:8]}.mp3"
        audio_path = self.generate_tts_audio(text, lang_code, filename)
        if audio_path and play_audio:
            self.play_audio(audio_path)
        return audio_path

    def preprocess_image(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            
            # Don't resize too small - keep more detail
            max_size = 768
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            return image
        except Exception as e:
            print(f"Error loading image: {e}")
            return None

    def process_image_to_speech(self, image_path, languages=None, save_audio=True, play_audio=True, return_base64_audio=True):
        results = {}
        if languages is None:
            languages = self.supported_languages

        image = self.preprocess_image(image_path)
        if image is None:
            return results

        print(f"\n🖼️ Processing image: {image_path}")
        print(f"📏 Image size: {image.size}")
        
        english_caption = self.generate_caption(image)
        print(f"\n📝 Final English Caption: {english_caption}")

        for lang in languages:
            print(f"\n🔄 Processing {lang}...")
            translation = self.translate_text(english_caption, lang)
            print(f"🌍 {lang.title()} Translation: {translation}")
            
            # Generate audio
            lang_code = self.lang_codes.get(lang.lower(), 'en')
            audio_path = None
            audio_base64 = None
            
            if return_base64_audio:
                # Generate base64 audio for API response
                audio_base64 = self.generate_tts_audio_base64(translation, lang_code)
            
            if save_audio or play_audio:
                # Generate file for local playback
                filename = f"{lang.lower()}_{uuid.uuid4().hex[:8]}.mp3"
                audio_path = self.generate_tts_audio(translation, lang_code, filename)
                
                if audio_path and play_audio:
                    self.play_audio(audio_path)
            
            results[lang] = {
                'text': translation, 
                'audio_path': audio_path,
                'audio_base64': audio_base64
            }

        return results


if __name__ == "__main__":
    print("Enhanced Multilingual Image-to-Speech Model Ready")
    model = MultilingualImageToSpeech()
    image_path = input("Enter image path: ").strip()
    if os.path.exists(image_path):
        model.process_image_to_speech(image_path, play_audio=True, return_base64_audio=False)
    else:
        print("Image file not found!")