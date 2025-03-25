import websockets
import asyncio
import whisper
import json
import logging
from pathlib import Path
from datetime import datetime
from colorama import Fore, Style, init
from typing import Optional, Dict, Any
from dataclasses import dataclass
import time
import psutil
import numpy as np

@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8765
    model_size: str = "medium"
    logs_dir: str = "logs"
    metrics_dir: str = "metrics"
    max_audio_size: int = 100 * 1024 * 1024  # 100MB
    log_metrics: bool = True
    performance_monitoring: bool = True

class TranscriptionServer:
    def __init__(self, config: Optional[ServerConfig] = None):
        self.config = config or ServerConfig()
        self.session_start = datetime.now()  # Move this up, before setup_directories
        self.transcription_count = 0
        self.performance_metrics = {
            'transcription_times': [],
            'audio_lengths': [],
            'gpu_usage': [],
            'memory_usage': []
        }
        self.setup_directories()  # Move this after session_start is set
        self.setup_logging()
    
        print(f"{Fore.MAGENTA}Initializing Whisper model...{Fore.WHITE}")
        try:
            self.model = whisper.load_model(self.config.model_size)
            print(f"{Fore.MAGENTA}Model loaded successfully~♪{Fore.WHITE}")
        except Exception as e:
            print(f"{Fore.RED}Model loading failed: {e}{Fore.WHITE}")
            raise

    def setup_directories(self):
        # Create organized directory structure
        timestamp = self.session_start.strftime("%Y%m%d_%H%M%S")
        self.session_dir = Path(f"sessions/session_{timestamp}")
        self.logs_dir = self.session_dir / "logs"
        self.metrics_dir = self.session_dir / "metrics"
        
        for directory in [self.session_dir, self.logs_dir, self.metrics_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def setup_logging(self):
        # Setup rotating file handler for logs
        log_file = self.logs_dir / "transcription.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("WhisperServer")

    async def log_metrics(self, audio_length: float, transcription_time: float):
        if not self.config.log_metrics:
            return

        metrics = {
            'timestamp': datetime.now().isoformat(),
            'audio_length': audio_length,
            'transcription_time': transcription_time,
            'transcriptions_per_second': 1/transcription_time if transcription_time > 0 else 0,
            'memory_usage': psutil.Process().memory_info().rss / 1024 / 1024,  # MB
            'cpu_percent': psutil.cpu_percent(),
        }

        # Save metrics to JSON file
        metrics_file = self.metrics_dir / f"metrics_{datetime.now().strftime('%Y%m%d')}.json"
        try:
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    existing_metrics = json.load(f)
                existing_metrics.append(metrics)
                metrics_data = existing_metrics
            else:
                metrics_data = [metrics]
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")

    async def handle_audio(self, websocket):
        try:
            while True:
                print(f"{Fore.MAGENTA}Waiting for audio input~{Fore.WHITE}")
        
                try:
                    audio_data = await asyncio.wait_for(websocket.recv(), timeout=30)
                    data = json.loads(audio_data)
            
                    # Convert the audio data to numpy array
                    audio_array = np.array(data['audio'], dtype=np.float32)
            
                    # Add some debug logging
                    self.logger.debug(f"Received audio array shape: {audio_array.shape}, dtype: {audio_array.dtype}")
            
                    # Process audio and measure performance
                    start_time = time.time()  # Add this line
                    audio_length = len(audio_array) / 16000
            
                    # Add more debug logging
                    self.logger.debug(f"Starting transcription of {audio_length:.2f}s audio")
            
                    result = self.model.transcribe(audio_array)
                    transcription_time = time.time() - start_time
            
                    # Log performance metrics
                    await self.log_metrics(audio_length, transcription_time)
            
                    response = {
                        "transcript": result["text"].strip(),
                        "confidence": result.get("confidence", 0),
                        "language": result.get("language", "unknown"),
                        "processing_time": transcription_time,
                        "timestamp": datetime.now().isoformat()
                    }
            
                    await websocket.send(json.dumps(response))
            
                    self.logger.info(
                        f"Transcription completed - Length: {audio_length:.2f}s, "
                        f"Processing time: {transcription_time:.2f}s, "
                        f"Language: {result.get('language', 'unknown')}"
                    )
            
                except asyncio.TimeoutError:
                    self.logger.warning("Connection timed out")
                    try:
                        await websocket.close(1001)  # Going away
                    except:
                        pass
                    break
                except json.JSONDecodeError as e:
                    self.logger.error(f"Invalid JSON format received: {e}")
                    await websocket.send(json.dumps({"error": "Invalid JSON format"}))
                except Exception as e:
                    self.logger.error(f"Transcription error: {str(e)}")
                    await websocket.send(json.dumps({"error": str(e)}))
                    break
            
        except websockets.exceptions.ConnectionClosed:
            self.logger.info("Client disconnected")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
        finally:
            # Make sure the connection is closed
            try:
                if not websocket.state.closed:  # Changed this line
                    await websocket.close()
            except:
                pass

    def generate_performance_report(self):
        """Generate a performance report for the current session"""
        if not self.performance_metrics['transcription_times']:
            return
        
        avg_time = np.mean(self.performance_metrics['transcription_times'])
        max_time = np.max(self.performance_metrics['transcription_times'])
        min_time = np.min(self.performance_metrics['transcription_times'])
        
        report = f"""
Performance Report ({self.session_start.strftime('%Y-%m-%d %H:%M:%S')})
================================================
Total Transcriptions: {self.transcription_count}
Average Processing Time: {avg_time:.2f}s
Max Processing Time: {max_time:.2f}s
Min Processing Time: {min_time:.2f}s
Average Audio Length: {np.mean(self.performance_metrics['audio_lengths']):.2f}s
        """
        
        report_file = self.session_dir / "session_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)

    async def start(self):
        banner_width = 56
        print(f"""
    {Fore.MAGENTA}╔{'═' * banner_width}╗
    ║ {Style.BRIGHT}~ Super-Awesome Whisper Transcription Server ~{' ' * (banner_width - 43)}║
    ║ Running on ws://{self.config.host}:{self.config.port}{' ' * (banner_width - 27 - len(str(self.config.port)))}║
    ║ Model: {self.config.model_size}{' ' * (banner_width - 8 - len(self.config.model_size))}║
    ║ Session: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}{' ' * (banner_width - 41)}║
    ╚{'═' * banner_width}╝{Fore.WHITE}
    """)
        try:
            async with websockets.serve(
                self.handle_audio, 
                self.config.host, 
                self.config.port,
                max_size=100 * 1024 * 1024,  # Increase to 100MB,
                ping_interval=20,
                ping_timeout=30,
                close_timeout=10
            ):
                await asyncio.Future()
        finally:
            self.generate_performance_report()

if __name__ == "__main__":
    config = ServerConfig(
        model_size="medium",
        log_metrics=True,
        performance_monitoring=True
    )
    server = TranscriptionServer(config)
    asyncio.run(server.start())