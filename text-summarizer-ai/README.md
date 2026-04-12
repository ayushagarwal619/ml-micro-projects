
# 🚀 Text Summarizer AI

An AI-powered web application that generates concise summaries from long-form text using a fine-tuned **T5 Transformer model**, built with a **FastAPI backend** and an interactive frontend UI.

---

## 🌟 Key Highlights

* 🧠 Fine-tuned **T5 model** for accurate summarization
* 🎯 Supports multiple summary lengths (Short / Medium / Long)
* ⚡ FastAPI backend for efficient local inference
* 🎨 Clean and responsive frontend UI
* 📊 Real-time summary generation
* 📁 Pre-trained model included for offline usage

---

## 🏗️ Project Structure

```bash
text-summarizer-ai/
│
├── app.py                  # FastAPI backend
├── index.html              # Frontend UI
├── README.md               # Documentation
│
├── saved_summary_model/    # Fine-tuned T5 model
├── Datasets/               # Training dataset
└── Results/                # Model outputs
```

---

## ⚙️ Tech Stack

* **Backend:** FastAPI
* **Model:** T5 (Transformer-based NLP model)
* **Frontend:** HTML, CSS, JavaScript
* **Libraries:** Transformers, PyTorch

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/text-summarizer-ai.git
cd text-summarizer-ai
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Run the Application

```bash
python -m uvicorn app:app --reload
```

---

### 5️⃣ Open in Browser

```
http://127.0.0.1:8000
```

---

## 🔌 API Endpoint

### 📌 Summarize Text

```
POST /summarize/
```

**Request Body:**

```json
{
  "dialogue": "Enter your text here",
  "length": "short | medium | long"
}
```

**Response:**

```json
{
  "summary": "Generated summary"
}
```

---

## 📊 Model Details

* Based on **T5 (Text-to-Text Transfer Transformer)**
* Fine-tuned on custom dataset
* Stored locally in `/saved_summary_model`
* Performs summarization without external APIs


---

## 💡 Future Improvements

* 🔊 Add Text-to-Speech (TTS) feature
* 🌐 Deploy full-stack version
* 📱 Improve UI/UX
* 🌍 Multi-language support

---

## 👨‍💻 Author

**Ayush Kumar Agarwal**

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!
