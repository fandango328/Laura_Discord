import os
import json
import queue
import numpy as np
import re
from vosk import Model, KaldiRecognizer

class VoskTranscriber:
    def __init__(self, model_path, sample_rate=16000):
        """Initialize Vosk transcriber with the specified model"""
        # Check if model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Vosk model not found at: {model_path}")
            
        # Load the model
        print(f"Loading Vosk model from {model_path}")
        self.model = Model(model_path)
        self.sample_rate = sample_rate
        
        # Create recognizer
        self.recognizer = KaldiRecognizer(self.model, sample_rate)
        self.recognizer.SetWords(True)  # Enable word timings
        
        # Initialize state
        self.reset()
        print("Vosk transcriber initialized successfully")
        
    def reset(self):
        """Reset the transcriber state"""
        self.complete_text = ""
        self.partial_text = ""
        self.previous_partial = ""
        self.buffer = []
        self.is_final = False
        self.silence_frames = 0
        self.last_confidence = 0
        self.segment_history = []
        
        # Reset recognizer for a new session
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
        self.recognizer.SetWords(True)
        
    def process_frame(self, frame_data):
        """
        Process a single audio frame and update transcription
        
        Args:
            frame_data: Audio frame as bytes or numpy array
            
        Returns:
            Tuple of (is_end_of_speech, is_speech_detected, partial_text)
        """
        # Convert numpy array to bytes if needed
        if isinstance(frame_data, np.ndarray):
            frame_bytes = frame_data.tobytes()
        elif isinstance(frame_data, list):
            frame_bytes = np.array(frame_data, dtype=np.int16).tobytes()
        else:
            frame_bytes = frame_data
            
        # Process the frame
        if self.recognizer.AcceptWaveform(frame_bytes):
            # We have a final result for a chunk of speech
            result = json.loads(self.recognizer.Result())
            text = result.get("text", "").strip()
            
            if text:
                # Check confidence of result
                confidence = 0
                words = result.get("result", [])
                if words:
                    confidences = [word.get("conf", 0) for word in words]
                    confidence = sum(confidences) / len(confidences) if confidences else 0
                
                # Store the segment with its confidence
                self.segment_history.append((text, confidence))
                
                # Rebuild complete text using all segments
                if self.segment_history:
                    # Use segments with good confidence
                    good_segments = [segment for segment, conf in self.segment_history if conf > 0.5]
                    if good_segments:
                        self.complete_text = " ".join(good_segments)
                    else:
                        # Fallback to all segments
                        self.complete_text = " ".join([segment for segment, _ in self.segment_history])
                
                # Clean up the text
                self.complete_text = self._normalize_text(self.complete_text)
                return False, True, self.complete_text
            return False, False, self.complete_text
        else:
            # We have a partial result
            partial = json.loads(self.recognizer.PartialResult())
            new_partial = partial.get("partial", "").strip()
            
            # Only update if there's an actual change
            if new_partial != self.previous_partial and new_partial:
                self.partial_text = new_partial
                self.previous_partial = new_partial
            
            # Detect if speech is happening based on partial text changing
            is_speech = bool(new_partial)
            
            return False, is_speech, self.partial_text
    
    def _normalize_text(self, text):
        """Clean up and normalize transcribed text"""
        if not text:
            return ""
            
        # Fix common issues
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()
        
        # Fix capitalization - capitalize first letter of sentences
        sentences = re.split(r'([.!?]\s+)', text)
        result = ""
        for i in range(0, len(sentences), 2):
            if i < len(sentences):
                sentence = sentences[i]
                if sentence and len(sentence) > 0:
                    sentence = sentence[0].upper() + sentence[1:]
                    result += sentence
                if i+1 < len(sentences):
                    result += sentences[i+1]
        
        # If no sentences detected, just capitalize first word
        if not result:
            if text:
                text = text[0].upper() + text[1:]
            return text
            
        return result
            
    def get_partial_text(self):
        """Get the current partial transcription"""
        return self._normalize_text(self.partial_text)
        
    def get_final_text(self):
        """Get the final complete transcription"""
        if not self.complete_text and self.partial_text:
            # If no final text but we have partial, use that
            return self._normalize_text(self.partial_text)
        return self._normalize_text(self.complete_text)
        
    def transcribe(self):
        """
        Get final transcription after processing is complete
        
        Returns:
            Transcribed text
        """
        # Finalize any remaining audio
        final_result = json.loads(self.recognizer.FinalResult())
        final_text = final_result.get("text", "").strip()
        
        if final_text:
            # Check confidence of result
            confidence = 0
            words = final_result.get("result", [])
            if words:
                confidences = [word.get("conf", 0) for word in words]
                confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Store with confidence
            self.segment_history.append((final_text, confidence))
            
            # Rebuild complete text using all segments
            good_segments = [segment for segment, conf in self.segment_history if conf > 0.5]
            if good_segments:
                self.complete_text = " ".join(good_segments)
            else:
                # Fallback to all segments
                self.complete_text = " ".join([segment for segment, _ in self.segment_history])
            
        # Clean up and normalize final text
        result = self._normalize_text(self.complete_text)
        
        # If we have nothing but have partial results, use those
        if not result and self.partial_text:
            result = self._normalize_text(self.partial_text)
            
        return result
        
    def cleanup(self):
        """Thoroughly cleanup VOSK resources"""
        # Force specific cleanup actions
        if hasattr(self, 'recognizer') and self.recognizer:
            self.recognizer = None  # Drop reference to recognizer
            
        if hasattr(self, 'model') and self.model:
            self.model = None  # Drop reference to model
            
        # Force garbage collection
        import gc
        gc.collect()
        print("VOSK resources released")
