# Third-party notices

## Constellation line data

`assets/constellations.lines.json` is derived from the d3-celestial project,
copyright (c) 2015 Olaf Frohn, and is redistributed under the BSD 3-Clause
License.  Source: <https://github.com/ofrohn/d3-celestial>.

The original BSD license terms are available in that project's `LICENSE` file.

## DINOv2 pretrained image encoder

The `meteor_event_transformer_boost_v1.pth` and
`meteor_event_transformer_boost_v2.pth` checkpoints include the unmodified
`facebook/dinov2-small` pretrained encoder (Meta Platforms, Inc. and affiliates),
distributed under the Apache License, Version 2.0. Only the event classification
head is trained on this application's reviewed meteor dataset.

Model source: <https://huggingface.co/facebook/dinov2-small>.
Original project: <https://github.com/facebookresearch/dinov2>.
The complete license is included at `assets/licenses/dinov2/LICENSE`.
