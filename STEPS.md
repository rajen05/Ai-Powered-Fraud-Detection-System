# Beginner Walkthrough: Credit Card Fraud Detection Project

This file assumes you've never done an ML project end-to-end before. Follow
it top to bottom. Every command is copy-pasteable. Expect this to take a
weekend (roughly 6-8 hours of focused work spread across Saturday/Sunday).

---

## Part 0: Before you start

You need:
- Python 3.10+ installed (`python3 --version` in your terminal to check)
- A free [Kaggle](https://www.kaggle.com) account
- A free [GitHub](https://github.com) account
- A code editor (VS Code recommended)

---

## Part 1: Project setup (15 min)

### 1.1 Create the project folder

You already have this scaffold. Open a terminal and navigate into it:

```bash
cd fraud-detection
```

### 1.2 Create a virtual environment

This keeps this project's Python packages separate from everything else on
your machine — standard practice, always do this.

```bash
python3 -m venv venv
```

Activate it:

- **Mac/Linux:** `source venv/bin/activate`
- **Windows:** `venv\Scripts\activate`

You'll know it worked because your terminal prompt now starts with `(venv)`.

### 1.3 Install the required packages

```bash
pip install -r requirements.txt
```

This installs pandas, scikit-learn, xgboost, imbalanced-learn, streamlit,
and everything else the project needs. Takes 1-3 minutes.

---

## Part 2: Get the dataset (10 min)

### 2.1 Download from Kaggle

1. Go to [kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Click **Download** (you may need to log in)
3. Unzip the download — you'll get a file called `creditcard.csv`

### 2.2 Place it in your project

Move `creditcard.csv` into the `data/` folder, so the path looks like:

```
fraud-detection/data/creditcard.csv
```

This file is ~150MB, which is why it's excluded from git (`.gitignore`
already handles this for you) — you never want to commit large data files
to a repo.

---

## Part 3: Explore the data (1-1.5 hrs)

### 3.1 Launch Jupyter

```bash
jupyter notebook
```

This opens a browser tab. Navigate into `notebooks/` and open
`01_eda_and_modeling.ipynb`.

### 3.2 Run the notebook cell by cell

Click each cell and press **Shift+Enter** to run it and move to the next
one. Don't skip ahead — run them in order top to bottom.

As you go, actually **look** at each chart before moving on:

- **Class distribution chart**: notice how tiny the fraud bar is compared
  to legit. This is the whole reason this project is interesting — say so
  in your own words somewhere (README, LinkedIn post, interview answer).
- **Amount distribution**: most transactions are small dollar amounts,
  with a long tail of big ones. This is why we look at the log-transformed
  version — it's easier to read.
- **Correlation chart**: some of the anonymized `V` columns correlate with
  fraud much more strongly than others. You won't know what they represent
  (that's intentional, for privacy) but you can still say "V14, V4, V12
  are the strongest fraud indicators" — that's a real, useful finding.

### 3.3 Understand what SMOTE is doing

Before running the SMOTE cell, read the comment above it. SMOTE creates
**synthetic** fraud examples by interpolating between real ones, only in
the training data. If you're fuzzy on why this matters, here's the
one-sentence version: if you balanced the data *before* splitting into
train/test, some of your "test" fraud examples would be synthetic copies
of training examples, and your model would look better than it really is.
That's called **data leakage**, and reviewers who understand ML will
specifically check whether you avoided this — it's a good thing to
explicitly point out that you handled correctly.

### 3.4 Run the three models

Each one (Logistic Regression, Random Forest, XGBoost) takes anywhere
from a few seconds to ~1 minute to train. XGBoost is usually the slowest
but also usually the best performer.

Read the printed output for each one. You'll see:
- **Precision**: of everything the model flagged as fraud, what % actually was fraud
- **Recall**: of all the actual fraud, what % did the model catch
- **F1**: the balance of the two
- **PR-AUC**: overall quality metric, most trustworthy one here

### 3.5 Fill in the results table

Copy the numbers from your notebook output into the results table in
`README.md`. This turns your README from a template into an actual report.

---

## Part 4: Understand the code behind the notebook (30 min)

The notebook calls functions from three files in `src/`. Open each one
and read through — you don't need to memorize them, just understand the
flow, because you'll likely get asked about this in an interview.

- **`src/preprocess.py`** — loading data, scaling `Amount`/`Time`, doing
  the train/test split, and the SMOTE function
- **`src/train.py`** — defines and trains all three models; running this
  file directly (`python src/train.py`) trains everything from scratch
  and saves the best model to `models/fraud_model.pkl`
- **`src/evaluate.py`** — the precision/recall/F1/PR-AUC scoring function

### 4.1 Run the training script directly (this is what the app needs)

```bash
python src/train.py
```

This re-trains all three models from the command line (not the notebook)
and saves whichever one scored best to `models/fraud_model.pkl`. The
Streamlit app in Part 5 needs this file to exist, so don't skip this step.

---

## Part 5: Run the Streamlit app (30-45 min)

### 5.1 Launch it

```bash
streamlit run app.py
```

This opens a browser tab at `localhost:8501`.

### 5.2 Try it out

- Click **"🎲 Load a random transaction"** to pull a real row from the
  dataset into the form, then click **"Check for fraud"**.
- Try a few different random transactions — including some that are
  actually labeled fraud (you won't know which without checking the data,
  but statistically ~1 in 580 will be).
- Drag the **threshold slider** left and right and watch how the
  flagged/not-flagged result can change. This is a live demonstration of
  the precision/recall tradeoff — a great thing to point out if you're
  demoing this in an interview.
- Click the **"📊 Model Insights"** tab to see the EDA charts rendered
  inside the app itself.

### 5.3 If something breaks

- **"No trained model found"** → you skipped `python src/train.py` in Part 4.4, go back and run it.
- **Import errors** → make sure your virtual environment is activated (`(venv)` should show in your terminal) and you ran `pip install -r requirements.txt`.
- **Port already in use** → run `streamlit run app.py --server.port 8502` instead.

---

## Part 6: Push to GitHub (20 min)

### 6.1 Initialize git (if you haven't already)

```bash
git init
git add .
git commit -m "Initial commit: fraud detection project"
```

Because of `.gitignore`, this will **not** upload `creditcard.csv` or the
trained model file — that's intentional, they're too large for git.

### 6.2 Create a GitHub repo

1. Go to [github.com/new](https://github.com/new)
2. Name it something clear, e.g. `credit-card-fraud-detection`
3. Don't initialize with a README (you already have one)
4. Click **Create repository**

### 6.3 Push your code

GitHub will show you the exact commands, but they'll look like:

```bash
git remote add origin https://github.com/YOUR_USERNAME/credit-card-fraud-detection.git
git branch -M main
git push -u origin main
```

---

## Part 7: Deploy the app (30-45 min)

### 7.1 Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
2. Click **New app**, pick your repo, branch `main`, main file `app.py`
3. Click **Deploy**

### 7.2 The data problem

Your deployed app won't have `creditcard.csv` (it's not in git, on
purpose). Two options:

- **Simplest for a portfolio piece:** the app already handles this
  gracefully — the "Model Insights" tab shows a message instead of charts,
  and the "load random transaction" button just won't appear. The manual
  entry form still works fully. This is fine for a demo.
- **More complete:** host a small sample CSV (a few thousand rows, not the
  full 285k) somewhere your app can reach it, or commit a trimmed sample
  file to the repo instead of the full dataset. Optional — don't let this
  block you from finishing.

You also need `models/fraud_model.pkl` to exist. Since it's in
`.gitignore`, you have two choices:
- Remove `models/*.pkl` from `.gitignore` and commit the trained model
  file directly (simplest — models this size are fine in git)
- Or add a step in your deploy process to train it fresh (more advanced, skip for now)

**Recommended for this weekend:** open `.gitignore`, delete the line
`models/*.pkl`, then `git add models/fraud_model.pkl`, commit, and push
again so the deployed app has the model it needs.

### 7.3 Get your live link

Once deployed, Streamlit gives you a URL like
`https://your-app-name.streamlit.app`. Add this to the top of your
`README.md` where it says "add your Streamlit Cloud link here."

---

## Part 8: Finish the writeup (30 min)

1. Fill in the results table in `README.md` with your actual numbers
2. Add a screenshot of the app (drag it into the README, or link an image)
3. Write 2-3 sentences in your own words about what you'd improve with
   more time (this shows self-awareness, always a good look) — SHAP
   explainability is a natural one to mention, see the stretch goal below

---

## Part 9 (optional stretch, if you have extra time): SHAP explainability

SHAP explains *why* a specific transaction was flagged, which is a strong
differentiator for a portfolio project. Rough idea, if you want to attempt
it after finishing everything above:

```python
import shap
explainer = shap.TreeExplainer(best_model)  # works for RF/XGBoost
shap_values = explainer.shap_values(X_test.iloc[:100])
shap.summary_plot(shap_values, X_test.iloc[:100])
```

This is genuinely optional — don't let it block you from shipping the
core project.

---

## You're done when:

- [ ] Notebook runs top to bottom without errors
- [ ] `python src/train.py` successfully saves a model
- [ ] Streamlit app runs locally and predictions work
- [ ] Code is pushed to GitHub
- [ ] App is deployed and the live link works
- [ ] README has real numbers filled in, not placeholders
- [ ] You can explain, out loud, in your own words: why accuracy is
      misleading here, what SMOTE does and why it's train-only, and why
      you picked your final model
