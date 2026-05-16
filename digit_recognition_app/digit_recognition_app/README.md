# 🔢 Handwritten Digit Recognition — MNIST
**CSC 309 Mini Project #5 | Okoro-Enyi Reginald | FUTO 2026**

A production-ready Streamlit web app that trains a CNN on MNIST and lets you upload (or draw) handwritten digits for real-time classification.

---

## Features
- ✅ Trains a 2-block CNN (~98.5% test accuracy) on first run, then caches the model
- ✅ Upload any image of a digit → get instant prediction + confidence chart
- ✅ Interactive HTML5 canvas to draw digits in-browser
- ✅ Training curves, benchmark comparison, random test sample viewer
- ✅ Dark-themed, polished UI built with Streamlit

---

## Run Locally

```bash
# 1. Clone / unzip the project
cd digit_recognition_app

# 2. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Deploy on Streamlit Community Cloud (Free, Recommended)

1. Push this folder to a **GitHub repository**
2. Go to https://share.streamlit.io → **New app**
3. Select your repo, branch, and set `app.py` as the main file
4. Click **Deploy** — it's live in ~2 minutes!

> **Note:** First load trains the CNN (~2 min on the cloud CPU). Subsequent loads use the cached model and are instant.

---

## Deploy on Render (Free tier available)

1. Push to GitHub
2. Go to https://render.com → **New Web Service**
3. Connect your repo
4. Set:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5. Deploy!

---

## Deploy on PythonAnywhere

1. Upload all files via the **Files** tab
2. Open a **Bash console** and run:
   ```bash
   pip3 install --user -r requirements.txt
   streamlit run app.py --server.port 8501
   ```
3. Set up a **Web App** with a custom domain or use the free subdomain

---

## Project Structure

```
digit_recognition_app/
├── app.py              ← Main Streamlit application
├── requirements.txt    ← Python dependencies
├── .streamlit/
│   └── config.toml     ← Streamlit theme & server config
└── README.md           ← This file
```

---

## CNN Architecture

| Layer | Output | Params |
|---|---|---|
| Input | 28×28×1 | — |
| Conv2D 32 filters + BN | 26×26×32 | 320 |
| MaxPool | 13×13×32 | 0 |
| Conv2D 64 filters + BN | 11×11×64 | 18,496 |
| MaxPool | 5×5×64 | 0 |
| Flatten | 1,600 | 0 |
| Dense 128 + Dropout 0.5 | 128 | 204,928 |
| Dense 10 Softmax | 10 | 1,290 |
| **Total** | | **~225,034** |

---

*Federal University of Technology Owerri · School of Computing · © 2026 Okoro-Enyi Reginald*
