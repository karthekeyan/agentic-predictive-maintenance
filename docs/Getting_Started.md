# Getting Started — First-Time Setup Guide

A complete guide for setting this project up on a computer that has never run it before — starting from nothing installed. Follow these steps in order.

---

## What you'll need accounts for

- A free **GitHub** account (to access the code)
- A free **Kaggle** account (to download the dataset)
- An **Anthropic API key** (for the AI reasoning — requires a small paid balance)

---

## Step 1: Install Git

Check if it's already installed:
```
git --version
```
If you see a version number, skip to Step 2.

If not: go to https://git-scm.com/download/win, download, run the installer accepting all defaults, then reopen Command Prompt and check again.

**One-time setup**, using your own name and GitHub email:
```
git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"
```

---

## Step 2: Install Python (via Anaconda)

Check if it's already installed:
```
python --version
```
If not: download and install Anaconda from https://www.anaconda.com/download — accept the default options.

---

## Step 3: Install Node.js

Check if it's already installed:
```
node --version
npm --version
```
If not: download the **LTS version** from https://nodejs.org, install with default options, then reopen Command Prompt and check again.

---

## Step 4: Install VS Code

Download from https://code.visualstudio.com/ and install with default options.

---

## Step 5: Clone the project

In Command Prompt, navigate to where you want the project to live, then:
```
git clone https://github.com/karthekeyan/agentic-predictive-maintenance.git
cd agentic-predictive-maintenance
```

Open the folder in VS Code:
```
code .
```

---

## Step 6: Install Python dependencies

```
pip install pandas numpy matplotlib jupyter chromadb anthropic python-dotenv langgraph fastapi uvicorn xgboost scikit-learn joblib kaggle
```

---

## Step 7: Download the real dataset (via Kaggle)

1. Create a free account at https://www.kaggle.com
2. Go to your Kaggle account settings → **API** section → **Create New Token** (or copy your API key if shown directly)
3. Set it up on your machine:
   ```
   setx KAGGLE_API_TOKEN "your-key-here"
   ```
   Close and reopen Command Prompt for this to take effect.
4. Download the dataset directly into the project:
   ```
   kaggle datasets download -d arnabbiswas1/microsoft-azure-predictive-maintenance -p data\raw --unzip
   ```
5. Confirm 5 CSV files now exist in `data\raw`:
   ```
   dir data\raw
   ```

---

## Step 8: Set up your Anthropic API key

1. Go to https://console.anthropic.com, create an account, and generate an API key under **API Keys**
2. In VS Code, create a new file at the project root named exactly `.env`
3. Add this line, replacing with your real key:
   ```
   ANTHROPIC_API_KEY=sk-ant-your-actual-key-here
   ```
4. Save. This file is already excluded from Git via `.gitignore` — it will never be uploaded anywhere.

---

## Step 9: Build the health scores and Knowledge Library

Open `notebooks/01_data_exploration.ipynb` in VS Code and run all cells from the top (this loads the data, calculates health scores, builds the MTBF statistics table, and populates the ChromaDB Knowledge Library with real historical cases).

---

## Step 10: Train the ML classifier

From the project root:
```
python src\tools\train_and_save_classifier.py
```
This will take a few minutes. Confirm it finishes with a classification report and saves three files into a new `models\` folder.

---

## Step 11: Set up the React dashboard

```
cd dashboard
npm install
cd ..
```

---

## Step 12: Run it

You now need **two terminal windows** running at the same time, plus a browser tab.

**Terminal 1:**
```
uvicorn backend_api:app --reload --port 8000
```

**Terminal 2:**
```
cd dashboard
npm run dev
```

**Browser:** open `http://localhost:5173`

---

## You're done

From here, see **`User_Guide.md`** for how to actually use the dashboard — reading the machine list, understanding each field, and interpreting results.

---

## If something goes wrong

| Symptom | Likely cause |
|---|---|
| `'git' is not recognized` | Git isn't installed or you need to reopen Command Prompt |
| `ModuleNotFoundError` for any Python package | Re-run the `pip install` command from Step 6 |
| Dashboard loads but the machine list stays empty | The backend (Terminal 1) isn't running — check for errors there |
| `FileNotFoundError` for the trained model | Step 10 (training the classifier) wasn't run, or wasn't run from the project root |
| Browser shows a CORS error | Confirm the backend is running on port 8000 and the frontend on 5173 — these are hardcoded to match each other |
