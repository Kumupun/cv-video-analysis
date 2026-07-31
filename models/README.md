# Model artifacts

The supplied AutoShot cut-detection checkpoint is included in this task bundle:

```text
models/
├── weights_for_cut.pth
├── SHA256SUMS
└── checkpoint_manifest.json
```

The checkpoint is loaded from `checkpoint["net"]` with `weights_only=True` and
must exactly match the bundled architecture in
`participant_4_cut_detection/autoshot_architecture.py`. The worker no longer
downloads Python architecture files at startup.

Verify the artifact before a real launch:

```bash
sha256sum -c models/SHA256SUMS
```

The tracking checkpoint was not supplied with this task. For fully offline
YOLO-World tracking, add:

```text
models/yolov8s-world.pt
```

Alternatively, set `TRACKING_MODEL_ID` to a valid Ultralytics model identifier
and allow that runtime to obtain the model. A local pinned file is preferred for
repeatable deployments.

Relevant settings:

- `CUT_MODEL_WEIGHTS_PATH` — defaults to `/models/weights_for_cut.pth`.
- `CUT_MODEL_THRESHOLD` — defaults to `0.55`.
- `TRACKING_MODEL_ID` — defaults to `/models/yolov8s-world.pt`.

`CUT_MODEL_ARCH_DIR` and `CUT_MODEL_ALLOW_DOWNLOAD` are retained only for
backward compatibility with older environment files; the current worker does
not use them.
