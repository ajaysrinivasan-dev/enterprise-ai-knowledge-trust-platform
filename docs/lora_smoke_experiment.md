# LoRA Smoke Experiment

**Status:** completed smoke test; not a production evaluation

## Configuration

- Model ID: `sshleifer/tiny-gpt2`
- Method: `lora`
- Dataset fingerprint: `ff4cf1b1d7ea520ffedc6e163e54e16f43fff0f9dd5f7f4a029d5dade46d230b2`
- Dataset status: synthetic smoke-test data only; not approved production data
- Split counts: train=5, validation=5, test=5
- Split seed: 2026
- Max length / batch size: 64 / 1
- Gradient accumulation: 1
- Learning rate / epochs: 0.0002 / 1.0
- Training steps: 2

## Verification and metrics

- Adapter reload: passed
- Held-out samples: 5
- Baseline: accuracy=0.2000, macro-F1=0.0667, weighted-F1=0.0667, latency=0.0003 ms
- Fine-tuned: accuracy=0.0000, macro-F1=0.0000, weighted-F1=0.0000, latency=19.4911 ms, failure rate=0.0000

## Limitations

- This was a synthetic smoke test with only five held-out examples.
- The run used two bounded training steps and does not establish model quality or production generalization.
- The fine-tuned result did not improve over the baseline and must not be interpreted as production evidence.
