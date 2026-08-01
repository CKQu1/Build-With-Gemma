# Deep Space Navigation Anomaly Detection

PyTorch framework for the Build with Gemma Track 2 idea in `pipeline.md`: detect anomalous spacecraft telemetry by forecasting future SMAP/MSL sensor values and flagging large prediction errors.

The repo is organized so a viewer can understand and run it even before downloading the Kaggle dataset. The default config trains on synthetic telemetry; the NASA configs point at the SMAP/MSL `.npy` files and `labeled_anomalies.csv`.

## What This Implements

This project is a small end-to-end system for spotting unusual behavior in spacecraft sensor data.

At a high level, it teaches a neural network what "normal" telemetry looks like. Once the network has learned normal patterns, it watches test data and asks: "Are the real sensor readings very different from what I expected?" Big differences are treated as possible anomalies.

The workflow is:

1. **Load telemetry data.**
   The code can read the NASA SMAP/MSL dataset, a custom CSV file, or built-in synthetic data for quick testing. Each row is one moment in time, and each column is a sensor or command-related feature.

2. **Clean the data.**
   If values are missing, the loader fills them in using interpolation, mean, or median imputation. This keeps small data gaps from breaking training.

3. **Normalize sensor values.**
   Sensor columns may use different ranges. The framework scales them so the model does not accidentally treat one sensor as more important just because its numbers are larger.

4. **Keep time in order.**
   The split is temporal, meaning earlier data is used for training and later data is used for validation or testing. This avoids letting the model peek into the future.

5. **Create short history windows.**
   Instead of feeding the full time series at once, the code creates many smaller examples. For example, the NASA config uses 100 past timesteps to predict the next 20 timesteps.

6. **Train a PyTorch model to predict what comes next.**
   The main model is `patch_tst`, a Transformer-style forecaster inspired by PatchTST. The repo also includes a regular Transformer and an LSTM autoencoder fallback.

7. **Compare predictions with reality.**
   During inference, the model predicts future sensor values. The code compares those predictions with the actual observed values. The larger the prediction error, the more unusual that moment looks.

8. **Choose an anomaly threshold.**
   The validation data gives a baseline for normal prediction errors. A high percentile of those errors becomes the cutoff. Test points above that cutoff are marked as anomalies.

9. **Save results for review.**
   The framework writes anomaly scores, summary metrics, trained model weights, config snapshots, and plots under `outputs/`.

10. **Turn detections into operational decisions.**
    The `new_scripts/` prototype adds a real-time handoff: streamed telemetry triggers a detector event, the mission agent looks up channel context such as `S-1`, and the output becomes a decision packet with affected crew roles, next checks, bounded-autonomy policy, and instructions.

## Repository Layout

```text
.
├── configs/
│   ├── default.yaml              # synthetic data, full local example
│   ├── smoke.yaml                # tiny fast check
│   ├── nasa_patchtst.yaml         # NASA SMAP/MSL Transformer config
│   └── lstm_autoencoder.yaml      # fallback model config
├── scripts/
│   ├── download_kaggle_dataset.py
│   ├── setup_venv.ps1
│   ├── setup_venv.sh
│   └── run_smoke_test.py
├── new_scripts/
│   ├── contracts.py             # shared event/decision contracts and S-1 context
│   ├── streamer.py              # tick-by-tick NASA telemetry streamer
│   ├── detector.py              # rolling z-score detector that emits Contract A
│   ├── agent.py                 # context-aware decision and crew-instruction layer
│   └── mission_control.py        # stream -> detect -> decide demo
├── src/deep_space_navigation/
│   ├── config.py
│   ├── data.py
│   ├── models.py
│   ├── train.py
│   ├── infer.py
│   └── visualize.py
├── tests/
├── pipeline.md
├── requirements.txt
├── environment.yml
└── pyproject.toml
```

## Setup With Conda

Create the environment:

```bash
conda env create -f environment.yml
conda activate gemma-space-anomaly
```

If you prefer a manual conda environment:

```bash
conda create -n gemma-space-anomaly python=3.11 -y
conda activate gemma-space-anomaly
pip install -r requirements.txt
pip install -e .
```

For a CUDA GPU, install the PyTorch build that matches your CUDA driver from the official PyTorch selector, then run the remaining installs.

## Setup With Python venv

If you do not use conda, create a local virtual environment from the project root.

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

You can also run the helper script:

```powershell
.\scripts\setup_venv.ps1
```

or:

```bash
bash scripts/setup_venv.sh
```

After activation, verify the install:

```bash
python -c "import torch; import deep_space_navigation; print(deep_space_navigation.__version__)"
```

## Live Judge Demo Notebook

For a smooth presentation, open:

```bash
jupyter notebook notebooks/judges_live_demo.ipynb
```

The notebook uses saved artifacts from `outputs/patchtst_nasa_p1` instead of retraining. It explains the trained PatchTST model, visualizes anomaly scores, and runs the live `S-1` stream-to-decision demo from `new_scripts/`.

## Quick Smoke Test

Run a tiny train-and-infer pass on synthetic telemetry:

```bash
python scripts/run_smoke_test.py
```

Or run the two steps yourself:

```bash
python -m deep_space_navigation.train --config configs/smoke.yaml
python -m deep_space_navigation.infer --checkpoint outputs/smoke/best_model.pt --make-plots
```

Expected outputs:

```text
outputs/smoke/
├── best_model.pt
├── history.json
├── resolved_config.yaml
├── anomaly_scores.csv
├── inference_summary.json
└── plots/
    ├── test_scores.png
    └── score_distribution.png
```

## Real-Time Mission Response Prototype

The `new_scripts/` folder contains the lightweight real-time framework used for the S-1 operational response demo.

Pipeline:

```text
streamer.py -> detector.py -> agent.py -> mission_control.py output
raw tick       Contract A      Contract B     crew-facing decision packet
```

The important contract files are:

- `new_scripts/contracts.py`: defines the detector event shape, the decision packet shape, crew roles, policy gates, and channel context registry.
- `new_scripts/agent.py`: reads a detection event, understands the channel context, decides who is affected, and builds crew instructions.
- `new_scripts/mission_control.py`: runs the end-to-end demo on `S-1`.

For `S-1`, the context registry treats the anonymized SMAP stream as a sensor/attitude telemetry channel. The known labeled anomaly window is `5300..5747`, and the playbook says to isolate the channel from automated estimator trust when a sharp spike exceeds `5 sigma`. If the anomaly persists beyond `50` timesteps, the agent escalates for ground review instead of blindly commanding high-risk actions.

Run the S-1 decision demo:

```bash
python new_scripts/mission_control.py
```

Expected behavior: the stream starts near the S-1 anomaly region, the detector emits a Contract A event around timestep `5538`, and the mission agent emits a Contract B packet similar to:

```text
event_id: evt_S1_5538
severity: medium
recommended_action: isolate_channel
policy_decision: AUTONOMOUS_ACT
command: isolate_channel
affected_roles: flight_director, adcs_officer, systems_engineer, science_payload_lead, ground_comms
```

The packet also includes targeted instructions. For example, the ADCS officer is told to cross-check `S-1` against `S-2`, `A-*`, and `D-*`; the science payload lead is told to flag products from the affected interval for pointing-quality review; ground communications is told to package the event for the next contact window.

Useful quick checks:

```bash
python new_scripts/contracts.py
python new_scripts/detector.py
python new_scripts/agent.py
```

If `data/raw/nasa/test/S-1.npy` is missing, download the NASA dataset first with the command in the next section.

## How to Read the Generated Plots

When inference runs with `--make-plots`, two diagnostic images are saved.

`test_scores.png` shows the model's behavior over time:

- The top panel shows the first sensor channel from the test data.
- Red points mark timesteps the model flagged as anomalies.
- Green points appear when ground-truth labels are available, such as with the NASA dataset.
- The bottom panel shows the anomaly score at each timestep.
- The dashed red line is the anomaly threshold. Scores above this line are flagged.
          
`score_distribution.png` shows the spread of anomaly scores:

- Most normal points should cluster toward the lower-score side.
- The dashed red line shows the selected cutoff.
- Points to the right of the cutoff are treated as unusual enough to investigate.

In plain language: the plots show where the model was surprised. A good result usually has low, steady scores for normal telemetry and clear spikes near true anomaly regions.

## NASA SMAP/MSL Dataset

Dataset: [`patrickfleith/nasa-anomaly-detection-dataset-smap-msl`](https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl) on Kaggle.

Download and organize it with:

```bash
python scripts/download_kaggle_dataset.py --output data/raw/nasa
```

The script uses KaggleHub, caches the archive under your user cache, and copies the training-ready files into:

```text
data/raw/nasa/
├── train/P-1.npy
├── test/P-1.npy
└── labeled_anomalies.csv
```

Then train and score channel `P-1`:

```bash
python -m deep_space_navigation.train --config configs/nasa_patchtst.yaml
python -m deep_space_navigation.infer --checkpoint outputs/patchtst_nasa_p1/best_model.pt --make-plots
```

To use a different channel, edit `data.channel_id` in the YAML config.

## Custom CSV Data

Set `data.source: csv` and provide `csv_path`. If `feature_cols` is omitted, all numeric columns except `timestamp_col` are used.

```yaml
data:
  source: csv
  csv_path: data/my_telemetry.csv
  timestamp_col: timestamp
  feature_cols: [sensor_a, sensor_b, sensor_c]
```

## Important Config Fields

- `data.sequence_length`: history window size. The NASA config follows the pipeline default of `100`.
- `data.prediction_horizon`: future steps to predict. The NASA config uses `20`.
- `data.stride`: sliding-window step size.
- `model.type`: `patch_tst`, `transformer`, or `lstm_autoencoder`.
- `detection.threshold_percentile`: percentile of validation scores used as the anomaly cutoff.
- `output.experiment_name`: subfolder under `outputs/`.

## Tests

Run:

```bash
python -m pytest
```

The tests cover sliding-window construction, score aggregation, model output shapes, and the S-1 mission-agent decision behavior.

## Notes

The NASA benchmark has official train/test files where labels are attached to the test telemetry. With `use_official_nasa_test: true`, this repo trains/validates on the official train stream and evaluates on the official test stream. For generic CSV or synthetic data, the framework uses the 70/15/15 temporal split described in `pipeline.md`.

## References

- NASA SMAP/MSL anomaly benchmark on Kaggle: https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl
- Original spacecraft telemetry anomaly work: https://arxiv.org/abs/1802.04431
