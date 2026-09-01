# Model weights

The trained YOLO11n checkpoint `best.pt` is intentionally not included in the GitHub repository because model weights are large.

Place the checkpoint in this directory:

```text
weights/best.pt
```

The web application will then load it automatically.

Alternatively, set the `MODEL_PATH` environment variable to the location of the checkpoint.
