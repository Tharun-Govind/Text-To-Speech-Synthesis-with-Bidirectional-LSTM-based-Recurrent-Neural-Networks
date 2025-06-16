from flask import Flask, request, render_template, send_file
import torch
import torchaudio
import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from io import BytesIO

from flask import Flask, render_template, request, send_file
from gtts import gTTS
import os

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    audio_file = None
    mel_spectrogram_file = None
    if request.method == "POST":
        text = request.form["text"]
        
        if text.strip():
            # Convert text to speech
            tts = gTTS(text, lang="en", slow=False)
            audio_file = "static/output.mp3"
            tts.save(audio_file)
              # Generate Mel spectrogram
            y, sr = librosa.load(audio_file, sr=None)  # Load audio
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # Convert to dB scale
            
            # Save Mel spectrogram as an image
            mel_spectrogram_file = "static/mel_spectrogram.png"
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(mel_spec_db, sr=sr, x_axis="time", y_axis="mel")
            plt.colorbar(format="%+2.0f dB")
            plt.title("Mel Spectrogram")
            plt.tight_layout()
            plt.savefig(mel_spectrogram_file)
            plt.close()

    return render_template("index.html", audio_file=audio_file, mel_spectrogram_file=mel_spectrogram_file)

        
    # return render_template("index.html", audio_file=audio_file)

@app.route("/download")
def download():
    return send_file("static/output.mp3", as_attachment=True)

# 
class BiLSTMTTS(torch.nn.Module):
    def __init__(self, input_dim=100, hidden_dim=128, output_dim=80, num_layers=2):
        super(BiLSTMTTS, self).__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim * 2, output_dim)  # BiLSTM has 2x hidden units

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out)

# Load model (Assuming pre-trained weights)
model = BiLSTMTTS()
# model.load_state_dict(torch.load("bilstm_tts.pth", map_location=torch.device("cpu")))
# model.eval()

def text_to_spectrogram(text):
    """ Converts input text to a Mel spectrogram """
    text_encoded = torch.rand(1, len(text), 100)  # Random embeddings (replace with proper text embedding)
    with torch.no_grad():
        mel_spectrogram = model(text_encoded).squeeze().numpy()
    return mel_spectrogram

def spectrogram_to_audio(mel_spec):
    """ Converts Mel spectrogram to audio using Griffin-Lim """
    mel_spec = np.exp(mel_spec)  # Inverse log-mel
    wav = librosa.feature.inverse.mel_to_audio(mel_spec, sr=22050, n_iter=32)
    return wav

@app.route("/x", methods=["GET", "POST"])
def index1():
    audio_path = None
    if request.method == "POST":
        text = request.form["text"]
        mel_spec = text_to_spectrogram(text)
        
        # Convert Spectrogram to Audio
        audio = spectrogram_to_audio(mel_spec)

        # Save as WAV file
        audio_buffer = BytesIO()
        sf.write(audio_buffer, audio, 22050, format="WAV")
        audio_buffer.seek(0)

        # Save spectrogram plot
        plt.figure(figsize=(10, 4))
        plt.imshow(mel_spec.T, aspect="auto", origin="lower", cmap="magma")
        plt.colorbar(label="Intensity")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.title("Generated Mel Spectrogram")
        plt.savefig("static/mel_spectrogram.png")
        plt.close()

        return send_file(audio_buffer, mimetype="audio/wav", as_attachment=True, download_name="output.wav")
    
    return render_template("index.html", audio_path=audio_path)

if __name__ == "__main__":
    app.run(debug=True)
