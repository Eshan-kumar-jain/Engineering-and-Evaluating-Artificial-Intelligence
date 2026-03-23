# Engineering and Evaluating Artificial Intelligence - Multi-Label Email Classifier

## 📌 Overview

This project implements a **modular, multi-label email classification system** for a multinational company's customer support tickets. It classifies support emails into hierarchical categories (Type 2 → Type 3 → Type 4) using a **Chained Multi-Output Classification** architecture built on software engineering best practices.

> **Course:** Engineering and Evaluating Artificial Intelligence (MSCAI1) - National College of Ireland  
> **Assessment:** Continuous Assessment 1 - Design Choice 1: Chained Multi-Outputs

---

## 🧩 Problem Statement

The company receives customer support emails (from AppGallery and Purchasing domains) that need to be classified across **multiple dependent label levels**:

| Level | Column | Example |
|-------|--------|---------|
| Type 1 | `y1` | Business group (AppGallery / Purchasing) - used for grouping, not classification |
| Type 2 | `y2` | Suggestion, Complaint, etc. |
| Type 3 | `y3` | Payment, Refund, etc. |
| Type 4 | `y4` | Subscription cancellation, etc. |

**The challenge:** Labels are hierarchically dependent - if Type 2 is predicted incorrectly, Types 3 and 4 are automatically wrong regardless of their individual prediction. A simple multi-class approach (classifying each type independently) fails to capture these dependencies.

---

## ✅ Solution: Chained Multi-Output Classification

Instead of classifying each type independently, we **chain** the labels into combined targets:

| Chained Column | What It Represents | Example Value |
|----------------|---------------------|---------------|
| `y_type2` | Type 2 alone | `Suggestion` |
| `y_type2_3` | Type 2 + Type 3 | `Suggestion__Payment` |
| `y_type2_3_4` | Type 2 + Type 3 + Type 4 | `Suggestion__Payment__Subscription cancellation` |

A single Random Forest model classifies each of these chained targets **per business group** (`y1`), enabling evaluation of the classifier's accuracy at each depth of the label hierarchy.

---

## 🏗️ Architecture & Design Principles

The project is built on three core software engineering principles:

### 1. Separation of Concerns (SoC)
Each stage of the pipeline is isolated into its own module:

```
main.py              → Controller / Orchestrator
preprocess.py        → Data loading, deduplication, noise removal
embeddings.py        → TF-IDF feature extraction
modelling/           → Model training, prediction & evaluation
model/               → ML model implementations (abstracted)
Config.py            → Shared configuration constants
```

### 2. Encapsulation (Data Modelling)
The `Data` class (`modelling/data_model.py`) encapsulates all input data (train/test splits, embeddings, labels) into a single object, ensuring a **consistent data interface** across all ML models.

### 3. Abstraction (Strategy Pattern)
`BaseModel` (`model/base.py`) defines an abstract interface with `train()`, `predict()`, `print_results()`, and `data_transform()` methods. All ML models (e.g., `RandomForest`) inherit from this base, ensuring:
- New models can be added without changing the controller
- All models are accessed through a **uniform interface**

---

## 📂 Project Structure

```
Engineering_And_Evaulating_Artificial_Intelligence/
│
├── main.py                  # Entry point - orchestrates the full pipeline
├── Config.py                # Shared configuration (column names, chained columns)
├── preprocess.py            # Data loading, deduplication, noise removal, translation
├── embeddings.py            # TF-IDF vectorization of text data
│
├── modelling/
│   ├── data_model.py        # Data class - encapsulates train/test splits
│   └── modelling.py         # Model orchestration - trains and evaluates models
│
├── model/
│   ├── __init__.py
│   ├── base.py              # Abstract base class for all ML models
│   └── randomforest.py      # Random Forest classifier implementation
│
└── data/
    ├── AppGallery.csv        # Support tickets - AppGallery domain
    └── Purchasing.csv        # Support tickets - Purchasing domain
```

---

## 🔄 Pipeline Flow

```
1. Load Data        → Read AppGallery.csv & Purchasing.csv, merge into single DataFrame
2. Preprocess       → De-duplicate email threads, remove noise (greetings, signatures, etc.)
3. Chain Labels     → Create y_type2, y_type2_3, y_type2_3_4 columns
4. Group by y1      → Split data by business group (AppGallery / Purchasing)
5. For each group, for each chained label:
   a. Embed         → Generate TF-IDF vectors (max 2000 features)
   b. Encapsulate   → Create Data object with train/test splits (80/20, stratified)
   c. Train         → Fit Random Forest (1000 estimators, balanced class weights)
   d. Predict       → Classify test set
   e. Evaluate      → Print classification report (precision, recall, F1-score)
```

---

## 🔑 What Changed from the Baseline (Activity 3)

| Aspect | Activity 3 (Before) | This Project (After) |
|--------|---------------------|----------------------|
| Classification | Multi-class only (`y2`) | Multi-label via chained outputs (`y2`, `y2+y3`, `y2+y3+y4`) |
| Config | Only `CLASS_COL` and `TYPE_COLS` | Added `CHAINED_COLS` for chained label definitions |
| Main Pipeline | Single classification loop per group | Nested loop: per group × per chained column |
| Label Handling | Classifies `y2` directly | Creates concatenated labels (e.g., `Suggestion__Payment`) |
| Output | One classification report per group | Three classification reports per group (one per label depth) |

---

## ⚙️ How to Run

### Prerequisites
- Python 3.10+
- Required packages:
  ```
  pip install pandas numpy scikit-learn
  ```
  Optional (for translation): `stanza`, `transformers`

### Execution
```bash
python main.py
```

The program will output classification reports for each business group at each label depth:
```
============================================================
Business Group: AppGallery
============================================================

--- Classifying: y_type2 ---
RandomForest
              precision    recall  f1-score   support
...

--- Classifying: y_type2_3 ---
RandomForest
...

--- Classifying: y_type2_3_4 ---
RandomForest
...
```


