* Added a generic model-based backend with the pinned official SuperMimic dynamics, BPTT, critic, replay, and update schedule.
* Added native training/playback entrypoints and differential reference tests; retained the earlier experimental backend for existing checkpoints.

* Added an explicit Adam optimizer and capture-safe cosine learning-rate schedule for published-paper experiments.
