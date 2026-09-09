# Mini Project — Insurance Prediction, Deep Learning, RAG & Scenario Assessment

UST — 100 Next Gen AI Engineer Cohort 3

| Part | Area | Core Task | Weight |
|---|---|---|---|
| A | Machine Learning | Insurance claim prediction model | 25% |
| B | Deep Learning | Rebuild/improve with a neural network | 25% |
| C | LangChain + RAG | M&A Playbook Q&A system | 30% |
| D | Scenario Assessment | Finance, Telecom & Healthcare cases | 20% |

## Files

| File | Covers |
|---|---|
| `MiniProject_PartA_PartB_Insurance.ipynb` | Part A (Tasks 1–5) and Part B (Tasks 7–10) |
| `MiniProject_PartC_MA_RAG.ipynb` | Part C (Tasks 12–16) |
| `MiniProject_PartD_Scenarios.md` | Part D — all 6 scenarios (renders on GitHub) |
| `MiniProject_PartD_Scenarios.pdf` | Part D as a formatted PDF report |
| `insurance2.csv` | Dataset for Parts A & B |
| `M&A Playbook_ Comprehensive Guide to Deals and Integration.pdf` | Knowledge source for Part C |

## How to run

### Parts A & B (Colab)

1. Open `MiniProject_PartA_PartB_Insurance.ipynb` in Google Colab.
2. Upload `insurance2.csv` to the session (it sits at `/content/`).
3. Runtime → Run all.

### Part C (Colab — needs a Gemini API key)

1. Open `MiniProject_PartC_MA_RAG.ipynb` in Google Colab.
2. Upload `M&A Playbook_ Comprehensive Guide to Deals and Integration.pdf` to the session.
3. Get a free API key from Google AI Studio (https://aistudio.google.com/apikey).
4. Runtime → Run all. The notebook prompts for the key at runtime via `getpass`, so the key is **never saved inside the notebook file**.

## Results summary

### Part A — Machine Learning

Dataset: 1,338 rows × 8 columns, no missing values. Target `insuranceclaim` is binary and mildly imbalanced (58.5% claim / 41.5% no-claim).

Preprocessing: `region` one-hot encoded (4 nominal levels); `sex` and `smoker` already binary. Continuous features scaled with `StandardScaler` **fitted on the training set only** to avoid data leakage.

| Model | Train Accuracy | Test Accuracy |
|---|---|---|
| Logistic Regression | 89.35% | 85.82% |
| Decision Tree (depth 4) | 87.66% | 87.31% |
| **Random Forest** | 100.00% | **95.90%** |

Random Forest is the best model. Its 100% training accuracy indicates memorisation, but test accuracy remains the highest. Top predictors: `bmi`, `children`, `charges`.

### Part B — Deep Learning

Feed-forward network (Keras): input 9 → Dense 16 (ReLU) → Dense 8 (ReLU) → Dense 1 (sigmoid), Adam + `binary_crossentropy`, 50 epochs with `validation_split=0.2`. Includes training-curve analysis, an overfitting diagnosis, and an improved version using `EarlyStopping` + `Dropout`.

**Conclusion:** deep learning does *not* beat Random Forest here. With only 1,338 rows of tabular data, tree ensembles remain the stronger choice — neural networks need far more data (or unstructured input) to justify their capacity.

### Part C — RAG

LangChain + Gemini + FAISS over the 25-page M&A Playbook only. Chunking experiment across five size/overlap combinations settled on `chunk_size=1000`, `chunk_overlap=200`. Every answer displays its retrieved source chunks and page numbers. The strict prompt is verified against out-of-scope questions (Microsoft share price, GDPR penalties), where the assistant correctly replies that the information is not available in the playbook instead of hallucinating.

### Part D — Scenarios

All six business cases answered: credit default prediction, financial document assistant, telecom churn, network operations copilot, patient readmission, and clinical knowledge assistant — covering problem formulation, technology selection, architecture, model selection, evaluation and business implications.
