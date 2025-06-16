# 🎤 Text To Speech Synthesis with Bidirectional LSTM Based Recurrent Neural Networks

This project is a Flask-based web application that converts text into speech using two powerful approaches:
- 🔊 **gTTS (Google Text-to-Speech)** – A fast and efficient API for converting text to audio.
- 🤖 **Bi-LSTM-based Recurrent Neural Network** – A deep learning model that generates Mel spectrograms from text, reconstructing audio using the Griffin-Lim algorithm.

## 🚀 Features
- Web interface to input text and listen to/download generated speech
- Dual TTS backends: `gTTS` and a custom Bi-LSTM model
- Mel spectrogram visualization using `Librosa` and `Matplotlib`
- Audio output in MP3 (gTTS) or WAV (Bi-LSTM) format
- Ready for experimentation with text embeddings and pre-trained models

---

## 🛠️ Tech Stack
- **Frontend**: HTML, CSS (embedded in Flask templates)
- **Backend**: Flask (Python)
- **AI/ML**: PyTorch (Bi-LSTM), Librosa, gTTS
- **Audio**: Soundfile, Matplotlib, Griffin-Lim algorithm
- **Deployment-ready** structure

---

## 📸 Demo

### gTTS Output
- Generates MP3 audio from text using Google's API
- Visualizes Mel spectrogram from audio

### Bi-LSTM Output
- Converts text to synthetic Mel spectrogram via Bi-LSTM
- Uses Griffin-Lim to convert spectrogram to WAV audio
- Spectrogram plotted and displayed

![Mel Spectrogram](static/mel_spectrogram.png)

---

## 🧠 How it Works

### 🔁 BiLSTM Model Architecture:
```python
class BiLSTMTTS(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=128, output_dim=80, num_layers=2):
        ...
````

> *Note*: Text is currently encoded using random embeddings. Replace with actual phoneme or character-level embeddings for production use.

---

## 🧪 Run Locally

1. **Clone the repo**

```bash
git clone https://github.com/Tharun-Govind/text-to-speech-bilstm.git
cd text-to-speech-bilstm
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the Flask app**

```bash
python app.py
```

4. **Open in Browser**

```
http://127.0.0.1:5000/
```

---

## 📂 Project Structure

```
├── app.py                  # Main Flask application
├── static/
│   └── mel_spectrogram.png  # Output spectrograms
├── templates/
│   └── index.html          # HTML frontend
```

---

## 📌 Future Enhancements

* Integrate Tacotron or transformer-based models
* Use real phoneme embeddings
* Add multilingual support
* Enable live waveform display

---

## 📃 License

This project is open-source and available under the MIT License.

---

### 👨‍💻 Developed by [Tharun Govind Sriramoju](https://www.linkedin.com/in/tharun-govind-sriramoju)

```
