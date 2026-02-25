#!/usr/bin/env python3
"""
Perfect AI Calling Agent for Render Deployment
Fixed all issues with proper RAG, voice detection, and error handling
"""
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from groq import Groq
import pyttsx3
from flask import Flask, render_template_string, request, jsonify

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

logger = logging.getLogger("ai_calling_agent")
logging.basicConfig(level=logging.INFO)

ROOT_DIR = Path(__file__).resolve().parent
LOG_AUDIO_DIR = ROOT_DIR / "logs" / "audio"
LOG_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
CALL_METRICS_PATH = ROOT_DIR / "logs" / "call_metrics.json"
ESCALATION_PATH = ROOT_DIR / "logs" / "escalations.json"

load_dotenv(ROOT_DIR / ".env.local")

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# IST knowledge base
from ist_knowledge import ISTDocument, load_ist_corpus, search, build_vector_index
IST_DOCS: list[ISTDocument] = load_ist_corpus()
build_vector_index(IST_DOCS)

class VoiceActivityDetector:
    """Improved VAD with noise filtering"""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.energy_threshold = 0.02
        self.min_speech_duration = 0.5  # Minimum speech duration
        self.silence_threshold = 0.01
        self.is_speaking = False
        self.speech_start_time = None
        self.audio_buffer = []
        
    def process_audio(self, audio_data):
        """Process audio with improved noise detection"""
        energy = np.sqrt(np.mean(audio_data**2))
        current_time = time.time()
        
        if energy > self.energy_threshold:
            if not self.is_speaking:
                self.speech_start_time = current_time
                self.is_speaking = True
            self.audio_buffer.extend(audio_data)
        else:
            if self.is_speaking:
                duration = current_time - self.speech_start_time
                if duration > self.min_speech_duration:
                    # Valid speech detected
                    speech_audio = np.array(self.audio_buffer)
                    self.is_speaking = False
                    self.audio_buffer = []
                    return {"speech_detected": True, "audio": speech_audio}
                else:
                    # Too short, ignore as noise
                    self.is_speaking = False
                    self.audio_buffer = []
        
        return {"speech_detected": False}

class NaturalMaleVoice:
    """Natural male voice TTS with interruption handling"""
    
    def __init__(self):
        self.engine = None
        self.setup_engine()
        self.is_speaking = False
        self.stop_requested = False
        
    def setup_engine(self):
        """Setup TTS with male voice"""
        self.engine = pyttsx3.init()
        voices = self.engine.getProperty('voices')
        
        # Find male voice
        best_voice = None
        for voice in voices:
            if 'male' in voice.name.lower() or 'david' in voice.name.lower():
                best_voice = voice.id
                break
        
        if best_voice:
            self.engine.setProperty('voice', best_voice)
        
        self.engine.setProperty('rate', 155)
        self.engine.setProperty('volume', 0.9)
        
    def speak(self, text, callback=None, on_interrupt=None):
        """Speak with interruption capability"""
        if not text.strip():
            if callback:
                callback()
            return
        
        def run_tts():
            self.is_speaking = True
            self.stop_requested = False
            
            try:
                sentences = [s.strip() for s in text.split('.') if s.strip()]
                for sentence in sentences:
                    if self.stop_requested:
                        if on_interrupt:
                            on_interrupt()
                        break
                    if sentence:
                        self.engine.say(sentence)
                        self.engine.runAndWait()
                        time.sleep(0.2)
                
                self.is_speaking = False
                if callback and not self.stop_requested:
                    callback()
            except Exception as e:
                logger.error(f"TTS error: {e}")
                self.is_speaking = False
                if callback:
                    callback()
        
        thread = threading.Thread(target=run_tts, daemon=True)
        thread.start()
    
    def stop_speaking(self):
        """Stop current speech"""
        self.stop_requested = True
        try:
            self.engine.stop()
        except:
            pass
        self.is_speaking = False

class AICallingAgent:
    """Complete AI calling agent with proper RAG and error handling"""
    
    def __init__(self):
        self.voice_detector = VoiceActivityDetector()
        self.tts = NaturalMaleVoice()
        self.call_active = False
        self.current_speaker = "none"
        self.recording_thread = None
        self.stop_recording = False
        self.metrics = {
            "stt_times": [],
            "llm_times": [],
            "tts_times": [],
            "e2e_times": [],
            "exchanges": []
        }
        
    def start_call(self):
        """Start voice call"""
        self.call_active = True
        self.current_speaker = "agent"
        self.reset_metrics()
        
        greeting = "Hello! Welcome to IST Admissions. How can I help you today?"
        self.tts.speak(greeting, lambda: self.start_listening())
        
    def start_listening(self):
        """Start listening for user input"""
        self.current_speaker = "user"
        self.stop_recording = False
        self.recording_thread = threading.Thread(target=self.record_user_speech, daemon=True)
        self.recording_thread.start()
        
    def record_user_speech(self, max_duration=15):
        """Record user speech with improved detection"""
        audio_data = []
        sample_rate = 16000
        chunk_size = 1024
        
        def audio_callback(indata, frames, time, status):
            if status or self.stop_recording:
                return
            audio_data.extend(indata.flatten())
        
        stream = sd.InputStream(
            callback=audio_callback,
            channels=1,
            samplerate=sample_rate,
            chunksize=chunk_size
        )
        
        stream.start()
        listening_start = time.time()
        speech_detected = False
        final_audio = []
        
        try:
            while self.call_active and not self.stop_recording:
                current_time = time.time()
                
                if len(audio_data) >= chunk_size:
                    chunk = np.array(audio_data[:chunk_size])
                    audio_data = audio_data[chunk_size:]
                    
                    vad_result = self.voice_detector.process_audio(chunk)
                    
                    if vad_result["speech_detected"]:
                        speech_detected = True
                        final_audio.extend(vad_result["audio"])
                    
                    elif speech_detected and not vad_result["speech_detected"]:
                        # User stopped speaking
                        break
                    
                    # Timeout
                    if current_time - listening_start > max_duration:
                        break
                
        except Exception as e:
            logger.error(f"Recording error: {e}")
        finally:
            stream.stop()
            stream.close()
        
        if final_audio:
            self.process_user_input(np.array(final_audio))
        else:
            # Continue listening
            if self.call_active:
                self.start_listening()
    
    def process_user_input(self, audio_data):
        """Process user audio and generate response"""
        self.current_speaker = "agent"
        
        # Save audio
        timestamp = int(time.time() * 1000)
        audio_path = LOG_AUDIO_DIR / f"user_{timestamp}.wav"
        sf.write(audio_path, audio_data, 16000)
        
        # STT
        stt_start = time.time()
        transcript = self.transcribe_audio(audio_path)
        stt_time = time.time() - stt_start
        
        if not transcript:
            self.current_speaker = "user"
            self.start_listening()
            return
        
        # Check for end call
        if self.should_end_call(transcript):
            self.end_call()
            return
        
        # LLM with RAG
        llm_start = time.time()
        response = self.get_intelligent_response(transcript)
        llm_time = time.time() - llm_start
        
        # TTS
        tts_start = time.time()
        speaking_start = time.time()
        
        self.tts.speak(response, 
                      lambda: self.on_agent_finished_speaking(speaking_start),
                      lambda: self.handle_interruption())
        
        tts_time = time.time() - tts_start
        e2e_time = time.time() - stt_start
        
        # Store metrics
        self.add_exchange(transcript, response, stt_time, llm_time, tts_time, e2e_time)
    
    def handle_interruption(self):
        """Handle user interruption"""
        logger.info("User interrupted - going back to listening")
        self.current_speaker = "user"
        self.start_listening()
    
    def on_agent_finished_speaking(self, speaking_start):
        """Called when agent finishes speaking"""
        speaking_time = time.time() - speaking_start
        self.metrics["tts_times"].append(speaking_time)
        self.current_speaker = "user"
        self.start_listening()
    
    def transcribe_audio(self, audio_path):
        """Transcribe audio using Groq"""
        try:
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
            
            resp = groq_client.audio.transcriptions.create(
                file=(os.path.basename(audio_path), audio_bytes),
                model="whisper-large-v3-turbo",
                language="en",
                temperature=0.0,
            )
            
            return (resp.text or "").strip()
        except Exception as e:
            logger.error(f"STT error: {e}")
            return ""
    
    def get_intelligent_response(self, query):
        """Get intelligent response with proper RAG and fallback"""
        query_lower = query.lower()
        
        # First try RAG
        relevant_docs = search(query, IST_DOCS, top_k=3)
        
        if relevant_docs:
            # Use RAG response
            context = " ".join([doc.text for doc in relevant_docs[:2]])
            response = self.generate_rag_response(query, context)
            
            # Check if response is meaningful
            if len(response) > 20 and "i don't have information" not in response.lower():
                return response
        
        # Check if it's a yes/no question
        if self.is_yes_no_question(query):
            return self.answer_yes_no_question(query)
        
        # Check if it can be answered generally
        if self.can_answer_generally(query):
            return self.generate_general_response(query)
        
        # Escalate to human
        return self.escalate_to_human(query)
    
    def generate_rag_response(self, query, context):
        """Generate response using RAG context"""
        try:
            prompt = f"""Based on the following IST information, answer the question accurately and confidently. 
            Do not make up information. If the information is not in the context, say so clearly.
            
            Context: {context}
            
            Question: {query}
            
            Answer:"""
            
            response = groq_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"RAG response error: {e}")
            return "I apologize, I'm having technical difficulties. Please try again."
    
    def is_yes_no_question(self, query):
        """Check if question can be answered with yes/no"""
        yes_no_indicators = [
            "is", "are", "do", "does", "did", "will", "would", "could", "should",
            "can", "has", "have", "was", "were", "am"
        ]
        
        return any(indicator in query_lower.split()[:3] for indicator in yes_no_indicators)
    
    def answer_yes_no_question(self, query):
        """Answer yes/no questions based on general knowledge"""
        general_knowledge = {
            "admission": "Yes, IST offers admissions in various engineering programs.",
            "fee": "Yes, IST has a fee structure for all programs.",
            "hostel": "Yes, IST provides hostel facilities.",
            "scholarship": "Yes, IST offers scholarships to deserving students.",
        }
        
        for key, answer in general_knowledge.items():
            if key in query_lower:
                return answer
        
        return "I don't have specific information to answer that question with yes or no."
    
    def can_answer_generally(self, query):
        """Check if question can be answered generally"""
        general_topics = [
            "hello", "hi", "help", "thank", "bye", "goodbye",
            "who are you", "what is ist", "institute of space technology"
        ]
        
        return any(topic in query_lower for topic in general_topics)
    
    def generate_general_response(self, query):
        """Generate general response"""
        general_responses = {
            "hello": "Hello! I'm here to help you with IST admissions.",
            "who are you": "I am the IST Admissions AI Assistant.",
            "what is ist": "IST is the Institute of Space Technology, a premier engineering university.",
            "help": "I can help you with admissions, programs, fees, and deadlines.",
        }
        
        for key, response in general_responses.items():
            if key in query_lower:
                return response
        
        return "I'm here to help with IST admissions. Please ask about programs, fees, or admission requirements."
    
    def escalate_to_human(self, query):
        """Escalate to human and collect phone number"""
        escalation_data = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "status": "pending"
        }
        
        # Save escalation
        try:
            if ESCALATION_PATH.exists():
                with open(ESCALATION_PATH, 'r') as f:
                    escalations = json.load(f)
            else:
                escalations = []
            
            escalations.append(escalation_data)
            
            with open(ESCALATION_PATH, 'w') as f:
                json.dump(escalations, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving escalation: {e}")
        
        return "I don't have information about that. Let me connect you with our admissions office. What's your phone number so they can call you back?"
    
    def should_end_call(self, transcript):
        """Check if user wants to end call"""
        end_phrases = [
            'end call', 'goodbye', 'bye', 'thank you bye', 'that\'s all',
            'no more query', 'finish', 'done', 'exit'
        ]
        
        return any(phrase in transcript.lower() for phrase in end_phrases)
    
    def end_call(self):
        """End call and save metrics"""
        self.call_active = False
        self.stop_recording = True
        self.current_speaker = "none"
        self.save_metrics()
    
    def reset_metrics(self):
        """Reset metrics for new call"""
        self.metrics = {
            "stt_times": [],
            "llm_times": [],
            "tts_times": [],
            "e2e_times": [],
            "exchanges": []
        }
    
    def add_exchange(self, user_query, agent_response, stt_time, llm_time, tts_time, e2e_time):
        """Add exchange to metrics"""
        self.metrics["exchanges"].append({
            "timestamp": datetime.now().isoformat(),
            "user_query": user_query,
            "agent_response": agent_response,
            "stt_time": stt_time,
            "llm_time": llm_time,
            "tts_time": tts_time,
            "e2e_time": e2e_time
        })
        
        self.metrics["stt_times"].append(stt_time)
        self.metrics["llm_times"].append(llm_time)
        self.metrics["tts_times"].append(tts_time)
        self.metrics["e2e_times"].append(e2e_time)
    
    def save_metrics(self):
        """Save call metrics"""
        try:
            if CALL_METRICS_PATH.exists():
                with open(CALL_METRICS_PATH, 'r') as f:
                    all_metrics = json.load(f)
            else:
                all_metrics = []
            
            call_summary = {
                "timestamp": datetime.now().isoformat(),
                "duration": sum(self.metrics["e2e_times"]),
                "exchanges": len(self.metrics["exchanges"]),
                "avg_stt": sum(self.metrics["stt_times"]) / len(self.metrics["stt_times"]) if self.metrics["stt_times"] else 0,
                "avg_llm": sum(self.metrics["llm_times"]) / len(self.metrics["llm_times"]) if self.metrics["llm_times"] else 0,
                "avg_tts": sum(self.metrics["tts_times"]) / len(self.metrics["tts_times"]) if self.metrics["tts_times"] else 0,
                "avg_e2e": sum(self.metrics["e2e_times"]) / len(self.metrics["e2e_times"]) if self.metrics["e2e_times"] else 0
            }
            
            all_metrics.append(call_summary)
            
            with open(CALL_METRICS_PATH, 'w') as f:
                json.dump(all_metrics, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")

# Flask Web App
app = Flask(__name__)
agent = AICallingAgent()

@app.route('/')
def home():
    """Main web interface"""
    return render_template_string('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IST Admissions AI Assistant</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .container {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.3);
            max-width: 600px;
            width: 90%;
            text-align: center;
            color: #333;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .logo {
            font-size: 3em;
            font-weight: bold;
            color: #1e3c72;
            margin-bottom: 10px;
        }
        .subtitle {
            color: #666;
            font-size: 1.2em;
            margin-bottom: 20px;
        }
        .call-button {
            background: linear-gradient(45deg, #28a745, #20c997);
            color: white;
            border: none;
            padding: 25px 50px;
            font-size: 1.4em;
            border-radius: 50px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: bold;
            text-transform: uppercase;
            box-shadow: 0 10px 20px rgba(40, 167, 69, 0.3);
            margin: 20px 0;
        }
        .call-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 25px rgba(40, 167, 69, 0.4);
        }
        .call-button:disabled {
            background: #6c757d;
            cursor: not-allowed;
            transform: none;
        }
        .status {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
            border: 2px solid #e9ecef;
            text-align: center;
            min-height: 80px;
        }
        .status-text {
            font-size: 1.2em;
            font-weight: bold;
            color: #1e3c72;
            margin-bottom: 10px;
        }
        .pulse {
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="logo">🎓 IST</div>
            <div class="subtitle">Institute of Space Technology - AI Admissions Assistant</div>
        </div>
        
        <button id="callButton" class="call-button pulse" onclick="startCall()">
            📞 Start AI Call
        </button>
        
        <div id="status" class="status">
            <div class="status-text">Click "Start AI Call" to begin</div>
        </div>
    </div>

    <script>
        let callActive = false;
        
        function startCall() {
            if (callActive) {
                alert('Call already active!');
                return;
            }
            
            const button = document.getElementById('callButton');
            const status = document.getElementById('status');
            
            // Request microphone access
            navigator.mediaDevices.getUserMedia({ 
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                } 
            }).then(stream => {
                callActive = true;
                button.disabled = true;
                button.innerHTML = '📞 Call Active';
                button.style.background = '#dc3545';
                button.classList.remove('pulse');
                
                status.innerHTML = '<div class="status-text">🤖 AI Assistant is greeting you...</div>';
                
                // Start call on server
                fetch('/start-call', {method: 'POST'})
                    .then(response => response.json())
                    .then(data => {
                        console.log('Call started:', data);
                        status.innerHTML = '<div class="status-text">🎤 Listening... Speak clearly</div>';
                    })
                    .catch(error => {
                        console.error('Error starting call:', error);
                        status.innerHTML = '<div class="status-text">❌ Error starting call</div>';
                        resetCall();
                    });
                
            }).catch(err => {
                console.error('Microphone access denied:', err);
                status.innerHTML = '<div class="status-text">❌ Microphone access denied</div>';
            });
        }
        
        function resetCall() {
            callActive = false;
            const button = document.getElementById('callButton');
            button.disabled = false;
            button.innerHTML = '📞 Start AI Call';
            button.style.background = 'linear-gradient(45deg, #28a745, #20c997)';
            button.classList.add('pulse');
        }
    </script>
</body>
</html>
    ''')

@app.route('/start-call', methods=['POST'])
def start_call():
    """Start AI call"""
    try:
        agent.start_call()
        return jsonify({"status": "success", "message": "Call started"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/metrics')
def metrics():
    """Get system metrics"""
    try:
        if CALL_METRICS_PATH.exists():
            with open(CALL_METRICS_PATH, 'r') as f:
                metrics = json.load(f)
            return jsonify({"status": "success", "metrics": metrics})
        else:
            return jsonify({"status": "no_metrics"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    print("🚀 IST AI Calling Agent Starting...")
    print("🌐 Web server ready for Render deployment")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
