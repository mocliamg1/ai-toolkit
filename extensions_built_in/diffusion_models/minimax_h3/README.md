# MiniMax-H3 training timesteps

MiniMax-H3 and MiniMax-H3 Ref2V can use different timestep distributions for
image and video batches in the same training run. In the training form, use
**Image Timestep Type** and **Video Timestep Type** beneath **Timestep Type**.
Both default to **Use Timestep Type**, preserving existing jobs.

For sigmoid image training with the default shifted video training, add these
fields to your existing job's `config.process[0].train` section:

```yaml
train:
  noise_scheduler: flowmatch
  timestep_type: shift
  image_timestep_type: sigmoid
  video_timestep_type: shift
```

Omit an override, set it to `null`, or choose **Use Timestep Type** to inherit
`timestep_type`. Overrides are selected from the target batch's frame count:
one frame uses the image setting; multiple frames use the video setting.
This also handles images in datasets with automatic video frame counts and
single-frame targets with reference images or videos.

The selected type controls timestep sampling and timestep-dependent loss
weighting. The UI offers `sigmoid`, `linear`, `shift`, and `weighted`; YAML also
accepts the existing advanced timestep types. Timestep Bias (`content_or_style`),
denoising bounds, and `first_timestep_chance` remain shared. These settings do
not change the inference/sample scheduler or the image/video dataset ratio.

Overrides require `noise_scheduler: flowmatch`. The legacy `linear_timesteps`
and `linear_timesteps2` flags must be disabled when using overrides so they
cannot silently replace the selected distributions.
