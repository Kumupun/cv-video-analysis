# Model artifacts

Both real-model checkpoints required by the `ml` profile are included:

```text
models/
├── weights_for_cut.pth
├── yoloe26l_military_assets.pt
├── SHA256SUMS
└── checkpoint_manifest.json
```

## AutoShot cut detection

`weights_for_cut.pth` is loaded from `checkpoint["net"]` with
`weights_only=True` and must exactly match the bundled architecture in
`participant_4_cut_detection/autoshot_architecture.py`. The worker does not
download Python architecture files at startup.

## YOLOE tracking detector

`yoloe26l_military_assets.pt` is the supplied custom YOLOE-26L detection
checkpoint. It contains its own fixed 12-class vocabulary:

- `camouflage_soldier`
- `weapon`
- `military_tank`
- `military_truck`
- `military_vehicle`
- `civilian`
- `soldier`
- `civilian_vehicle`
- `military_artillery`
- `trench`
- `military_aircraft`
- `military_warship`

The tracking adapter loads it with `ultralytics.YOLO` and then runs ByteTrack.
It does **not** call `set_classes()`, because doing that would replace the
vocabulary embeddings stored by the trained checkpoint. By default all 12
classes are tracked. `TRACKING_MODEL_CLASSES` may contain an exact subset of
the names above; unknown names fail during actor startup with a clear message.

Runtime downloads are disabled by default. The local file must be available at
`/models/yoloe26l_military_assets.pt`, which is provided by the read-only
`./models:/models:ro` Compose mount.

## Verification

Run before a real launch:

```bash
sha256sum -c models/SHA256SUMS
```

Relevant settings:

- `CUT_MODEL_WEIGHTS_PATH=/models/weights_for_cut.pth`
- `CUT_MODEL_THRESHOLD=0.55`
- `TRACKING_MODEL_ID=/models/yoloe26l_military_assets.pt`
- `TRACKING_MODEL_CLASSES=[]` to use all embedded classes
- `TRACKING_MODEL_IMAGE_SIZE=960`
- `TRACKING_MODEL_ALLOW_DOWNLOAD=false`
- `TRACKING_MODEL_CONFIDENCE=0.25`

The tracking checkpoint was saved by Ultralytics 8.4.112 and is loaded with the
same pinned package version. Ultralytics is AGPL-3.0; confirm that this license
and the training-dataset rights fit the intended deployment before production
use.
