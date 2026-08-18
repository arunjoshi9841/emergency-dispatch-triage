# Emergency Dispatch Triage

This project explores text classification for emergency-call transcripts. A shared DeBERTa encoder predicts the incident category, urgency level, and which response teams may be needed.

The project paper covers the implementation details and results [here](https://www.joshiarun.com/emergency_dispatch.pdf)

## What is included

- audio transcription and transcript cleanup
- synthetic transcript generation and labeling
- dataset preparation and stratified splitting
- shared and independent multi-task model variants
- training, evaluation, and plotting utilities
- a pipeline notebook showing how the pieces fit together

## Setup

Create a virtual environment, install the dependencies, and copy the environment template:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

The audio pipeline also requires `ffmpeg`. Add API credentials to `.env` only when using transcription or synthetic-data generation.

## Usage

The orchestration example is in `notebooks/pipeline.ipynb`. Model settings such as the encoder name, label definitions, and output paths are in `model/config/config.py`.

The repository does not include call recordings or training data. Supply data you are permitted to use and update the paths in the notebook before running the pipeline.

## Results

The code records per-task loss, accuracy, precision, recall, F1, and confusion matrices. No benchmark result is reported here because the training data and evaluated checkpoint are not part of this repository.

## Limitations

This is a research prototype, not a dispatch system. Synthetic labels can contain mistakes, and transcript models may miss context that is clear to a trained call taker. The output should not be used to make emergency-response decisions.

This project began as academic work; this repository contains only the implementation.
