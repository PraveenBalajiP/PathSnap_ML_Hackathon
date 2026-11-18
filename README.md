# PathSnap — GPS Map Matching to a Road Network

PathSnap maps noisy GPS observations to directed road-network arcs. Each trip includes a GPS trace (`.track`) and its road graph (`.nodes` and `.arcs`). The project creates candidate road arcs near each GPS observation, scores candidates with machine-learning models, and uses a Viterbi/HMM decoder to choose a route sequence that is plausible on the network.

This README describes the artifacts currently present in this project folder. The runnable demo uses the **student-test partition** because its predictions and held-out evaluation results are already available. It is not an evaluator-test submission.

## Current results

### Data volume and exploration

| Partition | Trips | GPS observations | Candidate feature rows | Road nodes | Directed arcs |
|---|---:|---:|---:|---:|---:|
| Training | 80 | 208,417 | 2,400,632 | 1,933,485 | 3,876,904 |
| Student test | 10 | 19,160 | 275,598 | 90,689 | 185,041 |

Exploration found **zero parse-issue rows**, missing/non-finite GPS rows, invalid coordinates, non-increasing timestamps, invalid arc endpoints, or isolated nodes in the summarized partitions. Positive GPS gaps over five seconds occurred 1,216 times in training and 44 times in student test. The road graphs can still contain disconnected components: the reported weak-component count ranged from 1 to 421 for training trips and 5 to 100 for student-test trips. Reference routes were available for all 80 training and all 10 student-test trips; invalid route arc IDs were zero. Consecutive route joins flagged as disconnected numbered 13 in training and one in student test.

For supervised training, 179,225 of 208,417 route-aligned training observations had the target arc among the generated candidates; 29,192 were candidate misses (about **14.0%**). The retained training candidate table therefore covers target-present observations. On student test, candidate misses were much lower: **1.62%** of evaluation-aligned observations.

### Model comparison

The five classifiers were compared with 5-fold **GroupKFold by trip**, using the 80 training trips and 2,400,632 candidate rows. `Observation top-1` measures how often the highest-scored candidate arc matches the target for an observation. Mean and standard deviation are across folds.

| Model | Observation top-1 | Average precision | ROC AUC | F1 at 0.5 | Log loss |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 60.68% ± 10.98 pp | 0.4280 | 0.9076 | 0.4071 | 0.3933 |
| Decision Tree | 64.06% ± 8.87 pp | 0.6631 | 0.8829 | 0.5212 | 0.3648 |
| MLP | 65.60% ± 8.43 pp | 0.7245 | 0.9434 | 0.6483 | 0.1294 |
| Random Forest | 66.46% ± 9.64 pp | 0.7355 | 0.9446 | 0.6459 | 0.1652 |
| XGBoost | 66.54% ± 9.11 pp | 0.7352 | 0.9429 | 0.5827 | 0.2065 |

XGBoost was selected for tuning because it had the best mean observation top-1 in the initial comparison; the difference from Random Forest was small. Random Forest had the highest initial average precision by a small margin. These fold metrics are model-selection estimates, not student-test accuracy.

### Bayesian XGBoost tuning

The Optuna study optimized mean observation top-1 using 5-fold GroupKFold by trip on training data. The saved study contains 14 trials, including pruned/failed trials. Its best was **trial 13**, with grouped-CV top-1 **66.789%** (fold values: 54.578%, 63.761%, 78.487%, 72.714%, 64.406%). This is a modest increase over the initial XGBoost mean of 66.538%; it is not a separate test-set result.

Best saved parameters:

| Parameter | Value |
|---|---:|
| `n_estimators` | 400 |
| `max_depth` | 9 |
| `learning_rate` | 0.03394 |
| `min_child_weight` | 4.19897 |
| `subsample` | 0.91104 |
| `colsample_bytree` | 0.64404 |
| `reg_lambda` | 4.22148 |
| `gamma` | 1.49925 |

The full-data tuned model fit used all 2,400,632 training candidate rows. Its saved student-test score table contains 275,598 candidate rows.

### Student-test HMM/Viterbi result

The reference-route alignment was used after decoding to create evaluation labels. On 19,160 aligned student-test observations:

| Measure | Classifier | XGBoost + Viterbi HMM |
|---|---:|---:|
| End-to-end top-1 | 70.54% | **78.60%** |
| Top-1 when target arc exists among candidates | 71.70% | **79.90%** |

The HMM improved end-to-end top-1 by **8.06 percentage points**. The candidate miss rate was **1.62%**: for those observations, the reference target arc was absent from the candidate set, so the decoder could not select it. The per-trip route summaries report all decoded transitions as reachable and zero unbridgeable transitions across the 10 trips. The UI displays the overall student-test metrics; they are not per-trip accuracy values.

## How the pipeline works

1. **Explore raw trip/network files.** Summarize GPS quality and timing, network size/connectivity, route availability, and data parsing issues.
2. **Preprocess and engineer features.** Project GPS and network geometry to metric coordinates, generate nearby directed-arc candidates, then calculate candidate distance/rank, projection fraction, arc length/bearing, GPS speed/heading/time deltas, and node in/out degrees.
3. **Train and compare candidate scorers.** Train Logistic Regression, Decision Tree, MLP, Random Forest, and XGBoost with folds grouped by trip.
4. **Tune XGBoost.** Bayesian/Optuna search selects parameters using training trips only, then refits and scores student-test candidates.
5. **Decode the route.** Viterbi combines candidate scores with road-network transition plausibility. The decoded arcs are expanded through shortest paths where possible.
6. **Evaluate student test.** Reference `.route` files are used after prediction to align held-out evaluation targets. They are not included in model features or the API response.
7. **Serve the demo.** FastAPI reads trip geometry and precomputed HMM predictions; the static frontend draws GPS points and predicted arcs.

## Project layout

```text
PathSnap_ML_Hackathon/
├── train/                         # 80 training trip folders
├── student_test/                  # 10 held-out development trip folders
├── evaluator_test/                # Separate evaluation partition; not used in current demo
├── exploration_outputs/            # GPS, network, route, and parse summaries
├── processed_datasets/             # Candidate features, observations, labels, manifest
├── model_results/
│   ├── logistic_regression/
│   ├── decision_tree/
│   ├── MLP/
│   ├── random_forest/
│   ├── XGBoost/                     # CV, Bayesian study, tuned model, student scores
│   └── hmm_viterbi/                 # Predictions, metrics, per-trip summaries, routes
├── backend/app/main.py              # FastAPI API
├── backend/requirements.txt         # API dependencies
├── backend/requirements-ml.txt      # ML/training dependencies
├── frontend/                        # Static HTML/CSS/JavaScript UI
├── dataset_exploration.ipynb
├── data_preprocessing_cum_feature_engineering.ipynb
└── model_training_experiments.ipynb
```

### Key result artifacts

- `exploration_outputs/gps_summary.csv`, `network_summary.csv`, `route_summary.csv`, `parse_issues.csv`
- `processed_datasets/trip_manifest.csv` and `.json`
- `processed_datasets/training_candidates.csv` and `student_test_candidates.csv`
- `processed_datasets/student_test_eval_labels.csv` (evaluation labels generated separately after prediction)
- `model_results/<model>/<model>_cv_summary.csv` and `*_cv_folds.csv`
- `model_results/XGBoost/xgboost_bayesian_best.json`, `xgboost_bayesian_trials.csv`, `xgboost_tuned_model.joblib`, and `xgboost_tuned_student_test_candidate_scores.csv`
- `model_results/hmm_viterbi/hmm_viterbi_student_test_predictions.csv`, `hmm_viterbi_student_test_metrics.csv`, `hmm_viterbi_summary.csv`, and `hmm_viterbi_route_sequences.json`

The current data tables are uncompressed CSVs. The `.joblib` files are serialized scikit-learn models; the Optuna `.db` is its study store.

## Run the frontend and backend on separate laptops

Both laptops must be on the same reachable LAN. The backend laptop hosts the API; the frontend laptop hosts the static UI. In PowerShell, run `ipconfig` on each laptop and note its active Wi-Fi/Ethernet IPv4 address. Do not use the VMware, WSL, Bluetooth, or loopback addresses.

### Backend laptop

From the project root, create/use the virtual environment and install dependencies if needed:

```powershell
cd D:\Projects\PathSnap_ML_Hackathon
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
```

Set CORS to the **frontend laptop’s UI origin**. For example, if the frontend laptop is `192.168.137.1` and serves port 5500:

```powershell
$env:PATHSNAP_ROOT = 'D:\Projects\PathSnap_ML_Hackathon'
$env:PATHSNAP_PARTITION = 'student_test'
$env:PATHSNAP_RESULTS_DIR = 'D:\Projects\PathSnap_ML_Hackathon\model_results'
$env:PATHSNAP_UI_ORIGINS = 'http://192.168.137.1:5500'
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

Keep this terminal open. The API binds to all network interfaces. Check `http://<BACKEND-IP>:8000/api/health`; it should report `prediction_file_exists: true`. The trip list is at `/api/trips` and interactive docs at `/docs`.

### Frontend laptop

Copy the `frontend` folder to the frontend laptop. From that folder’s parent directory, run:

```powershell
python -m http.server 5500 --bind 0.0.0.0 --directory frontend
```

Open `http://<FRONTEND-IP>:5500` in that laptop’s browser. In **Backend API address**, enter `http://<BACKEND-IP>:8000` (the backend laptop’s IP). The `PATHSNAP_UI_ORIGINS` value on the backend must exactly match the frontend page origin, including scheme and port. If requests still return CORS 400, check the browser’s Network tab for the `Origin` request header and use that exact value. Allow inbound TCP port 8000 on the backend laptop and 5500 on the frontend laptop in Windows Firewall for the private network.

## Important interpretation and limitations

- The 2.4 million training rows are **candidate-arc rows**, not 2.4 million distinct GPS fixes. Training comprises 208,417 GPS observations across 80 trips.
- In the current supervised training table, candidate rows are retained for aligned training observations whose target arc is present in the candidate set. As a result, grouped-CV top-1 evaluates candidate ranking on retained examples; it does not measure candidate-generation misses. The student-test end-to-end score does include those misses.
- Grouping CV by trip reduces leakage between a trip’s candidate rows, but fold standard deviations are still substantial. The Bayesian best CV result is a tuning estimate, not an independent test score.
- Student-test labels are route-aligned evaluation targets. They support development comparison but should not be represented as evaluator-test performance.
- Candidate misses cap achievable accuracy: the decoder cannot select a true arc missing from the candidate list.
- The current frontend/backend preview uses precomputed student-test predictions. The API does not reconstruct candidates or run XGBoost/Viterbi on every request; it returns the saved HMM result for the selected trip.
- `evaluator_test/` exists in the project, but no evaluator-test candidate inference, predictions, or results are reported here. Do not claim evaluator accuracy from the student-test metrics.
- The root-level Bayesian and HMM scripts in this snapshot still contain paths for the earlier `datasets/processed_v1` layout. The saved results above are the verified artifacts for this project snapshot; update script paths before using those scripts to regenerate outputs in the current top-level `processed_datasets/` and `model_results/` layout.
