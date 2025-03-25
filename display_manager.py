import pygame
import asyncio
import random
import time
from pathlib import Path

class DisplayManager:
    def __init__(self):
        pygame.init()
        self.base_path = Path('/home/user/LAURA/pygame')
        self.screen = None
        self.image_cache = {}
        self.current_state = 'sleep'
        self.current_mood = 'neutral'
        self.last_state = None
        self.last_mood = None
        self.current_image = None
        self.last_image_change = None
        self.state_entry_time = None
        self.initialized = False
        self.moods = [
            "neutral",
            "happy", 
            "confused",
            "disappointed",
            "annoyed",
            "surprised"
        ]
        self.states = {
            'listening': str(self.base_path / 'listening'),
            'idle': str(self.base_path / 'idle'),
            'sleep': str(self.base_path / 'sleep'),
            'speaking': str(self.base_path / 'speaking'),
            'thinking': str(self.base_path / 'thinking'),
            'tools': str(self.base_path / 'tools'),
            'wake': str(self.base_path / 'wake')
        }
        self.setup_display()
        self.load_image_directories()
        
        # Do proper initial display
        if 'sleep' in self.image_cache:
            self.current_image = random.choice(self.image_cache['sleep'])
            self.screen.blit(self.current_image, (0, 0))
            pygame.display.flip()
            # Set timings AFTER first image is displayed
            self.last_image_change = time.time()
            self.state_entry_time = time.time()
            self.initialized = True
        else:
            raise RuntimeError("Failed to load sleep images")
        
    def setup_display(self):
        self.screen = pygame.display.set_mode((512, 512))
        pygame.display.set_caption("LAURA")
        
    def load_image_directories(self):
        #print("\nLoading image directories...")
        for state, directory in self.states.items():
            #print(f"\nChecking state: {state}")
            #print(f"Path: {directory}")
            
            if state == 'speaking':
                self.image_cache[state] = {}
                for mood in self.moods:
                    mood_path = Path(directory) / mood
                    print(f"Checking mood: {mood}")
                    #print(f"Path: {mood_path}")
                    if mood_path.exists():
                        png_files = list(mood_path.glob('*.png'))
                        #print(f"Found {len(png_files)} images")
                        if png_files:
                            self.image_cache[state][mood] = [
                                pygame.transform.scale(pygame.image.load(str(img)), (512, 512))
                                for img in png_files
                            ]
            else:
                state_path = Path(directory)
                if state_path.exists():
                    png_files = list(state_path.glob('*.png'))
                    #print(f"Found {len(png_files)} images")
                    if png_files:
                        self.image_cache[state] = [
                            pygame.transform.scale(pygame.image.load(str(img)), (512, 512))
                            for img in png_files
                        ]

    async def update_display(self, state, mood=None):
        # Wait for initialization
        while not self.initialized:
            await asyncio.sleep(0.1)
            
        if mood is None:
            mood = self.current_mood

        #print(f"\nUpdating display:")
        #print(f"Current state: {self.current_state} -> {state}")
        #print(f"Current mood: {self.current_mood} -> {mood}")
        
        try:
            # Update state tracking variables
            self.last_state = self.current_state
            self.current_state = state
            self.current_mood = mood
            
            # Handle image selection
            if state == 'speaking':
                if mood not in self.image_cache[state]:
                    print(f"Warning: Invalid mood '{mood}', using neutral mood")
                    mood = 'neutral'
                self.current_image = random.choice(self.image_cache[state][mood])
            elif state in self.image_cache:
                self.current_image = random.choice(self.image_cache[state])
            else:
                print(f"Error: Invalid state '{state}'")
                return
                
            # Display the image
            self.screen.blit(self.current_image, (0, 0))
            pygame.display.flip()
            
            # Only update rotation timer for idle/sleep states
            if state in ['idle', 'sleep']:
                self.last_image_change = time.time()
                self.state_entry_time = time.time()
                
            print(f"Display updated - State: {self.current_state}, Mood: {self.current_mood}")
                
        except Exception as e:
            print(f"Error updating display: {e}")

    async def rotate_background(self):
        while not self.initialized:
            await asyncio.sleep(0.1)
        #print("Rotation background task started")
    
        while True:
            try:
                current_time = time.time()
                #print(f"\nRotation check at: {time.strftime('%H:%M:%S')}")
            
                if self.current_state in ['idle', 'sleep']:
                    time_diff = current_time - self.last_image_change
                    #print(f"Current state: {self.current_state}")
                    #print(f"Time since last change: {time_diff:.2f} seconds")
                
                    if time_diff >= 15:
                        available_images = self.image_cache[self.current_state]
                        if len(available_images) > 1:
                            current_options = [img for img in available_images if img != self.current_image]
                            if current_options:
                                new_image = random.choice(current_options)
                                self.current_image = new_image
                                self.screen.blit(self.current_image, (0, 0))
                                pygame.display.flip()
                                self.last_image_change = current_time
                                #print(f"Rotated image at {time.strftime('%H:%M:%S')}")
                    #else:
                        #print(f"Not time to rotate yet. Waiting {15 - time_diff:.2f} more seconds")
                #else:
                    #print(f"Not in rotating state. Current state: {self.current_state}")
                
            except Exception as e:
                print(f"Error in rotate_background: {e}")
                traceback.print_exc()
        
            # Instead of a long sleep, do shorter sleeps
            for _ in range(15):  # 15 one-second sleeps
                await asyncio.sleep(1)
                # Add a check to see if we're still running
                print(".", end="", flush=True)
            
    def cleanup(self):
        pygame.quit()
