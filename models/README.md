# Model artifacts

AutoShot is bundled. The real ML profile now uses a two-detector ensemble:

```text
models/
├── weights_for_cut.pth
├── best.pt
├── yoloe26l_military_assets.pt
├── SHA256SUMS
└── checkpoint_manifest.json
```

## AutoShot cut detection

`weights_for_cut.pth` is loaded from `checkpoint["net"]` with
`weights_only=True` and must exactly match the bundled architecture in
`participant_4_cut_detection/autoshot_architecture.py`. The worker does not
download Python architecture files at startup. Its one-hot head produces
point `hard_cut` events and its many-hot head produces inclusive
`gradual_transition` ranges. A gradual range must also contain a confident
one-hot midpoint prediction; this prevents the many-hot head alone from turning
motion or lighting bursts into scene transitions. The frontend shows gradual
ranges as yellow bands and the final JSON includes their start/end frames,
timestamps, and confidence.

## Detection ensemble and tracking

`best.pt` is the friend's fine-tuned YOLO-World checkpoint and is configured as
the primary detector. The runtime verifies that it loads as an Ultralytics
`WorldModel`; task-selected frontend classes are installed with
`model.set_classes()`.

`yoloe26l_military_assets.pt` is the auxiliary fine-tuned detector. Its
embedded class names must match the corresponding names used by `best.pt`, for
example `camouflage_soldier`, `weapon`, and `military_tank`. Auxiliary classes
which were not selected for the task are ignored.

Both models receive the same frame. Their boxes are mapped to the primary
class vocabulary and merged with class-aware NMS. The merged boxes are then
passed to one ByteTrack instance, so duplicate detections do not create two
independent track ID sequences. ByteTrack is reset after every AutoShot scene
transition as before.

The two detector calls run concurrently by default. On a GPU that cannot hold
both models' inference activations, set `TRACKING_PARALLEL_INFERENCE=false` to
run the same ensemble sequentially (both model weights still need to fit).

Runtime downloads are disabled by default. Both local files are provided by
the read-only `./models:/models:ro` Compose mount.

The ML image installs the official OpenAI CLIP source from an immutable commit
and preloads its SHA-verified `ViT-B/32` text encoder during `docker build`.
`set_classes()` therefore has the required `clip` module and cached text
weights when the Ray actor starts.

## Verification

If the checksums file contains the model supplied by the friend, run:

```bash
sha256sum -c models/SHA256SUMS
```

Relevant settings:

- `TRACKING_MODEL_ID=/models/best.pt`
- `TRACKING_ENSEMBLE_MODEL_ID=/models/yoloe26l_military_assets.pt`
- `TRACKING_ENSEMBLE_IOU_THRESHOLD=0.55`
- `TRACKING_PARALLEL_INFERENCE=true`
- `TRACKING_MODEL_CLASSES=[]` to use the checkpoint's embedded vocabulary when
  a request does not specify classes
- `TRACKING_MODEL_IMAGE_SIZE=960`
- `TRACKING_MODEL_ALLOW_DOWNLOAD=false`
- `TRACKING_MODEL_CONFIDENCE=0.50`
- `TRACKING_TRACK_LOW_THRESHOLD=0.10`
- `TRACKING_NEW_TRACK_THRESHOLD=0.55`

Ultralytics is pinned to `8.4.112` and licensed under AGPL-3.0. Confirm that its
license and both checkpoint terms fit the intended deployment before
production use.
