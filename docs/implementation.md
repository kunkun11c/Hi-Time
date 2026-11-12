# Implementation

Inputs use `(batch, variables, time)`. The model has three stages, corresponding to Sections 4.1–4.3 of *Hi-Time: Hierarchical Latent Prediction for Multivariate Time Series Classification*.

## Temporal representation pre-training

For each sample, the encoder computes an FFT, averages amplitudes across variables, excludes the DC bin, and selects the largest S frequency components. The downsampling step for a selected frequency f is `round(max(selected_f) / f)`, clamped to the sequence length. Average pooling performs the downsampling; sequences shorter than one patch are padded by repeating the final value.

All scales share one PatchTST backbone. Their patch features are aligned to the original-resolution patch count by adaptive average pooling, passed through separate scale MLPs, and summed (Eqs. 1–3). In the configuration, the length of `scales` determines S; its entries are retained for compatibility and are not fixed downsampling steps.

Self-attention treats variables as tokens independently at each patch and adds a residual (Eqs. 4–6). A two-layer GELU MLP supplies patch scores. Softmax over patches, a weighted patch sum, and a mean over variables produce the final representation (Eqs. 7–9). The encoder is trained for classification, then frozen.

## Residual quantization

K-Means is fitted to the frozen encoder's training-set representations. Each level assigns the nearest centroid and passes the remaining residual to the next level (Eqs. 10–12). Codebooks remain fixed afterward. Fitting uses K-Means++, ten initializations, and `stage2_epochs` as the maximum number of Lloyd iterations. If K exceeds the number of training examples, the fitted centers are repeated to fill the configured vocabulary.

## Hierarchical latent prediction

Each raw time point is projected from the variable dimension to GPT-2's hidden dimension. GPT-2 adds its own positional embeddings once, and its last token supplies the global context (Eq. 13).

At level l, an MLP takes this context and the embeddings of codes predicted at levels before l. The highest-scoring code is selected by hard argmax and looked up in that level's learnable embedding table. The classifier receives the context and all predicted code embeddings (Eqs. 14–15). Training and inference use this same prediction path. Target codes and class labels are never predictor inputs.

The loss is:

```text
sum_l CE(predicted_code_l, target_code_l) + λ * CE(predicted_class, label)
```

Code losses are summed, with no division by the number of levels (Eq. 16). Since argmax is discrete, later losses do not propagate through a selected code index into an earlier level's logits. The embedding tables still receive gradients from the losses that consume them.

## Changes and unspecified details

The earlier release fed quantized target embeddings to GPT-2 and read code predictions at positions containing their own targets. It also used a true-class condition in the auxiliary branch. These paths have been replaced by raw-series input and sequential prediction. The encoder now uses FFT-derived scales, shared backbone weights, additive scale fusion, and the gating equations above; residual Lloyd K-Means replaces minibatch EMA quantization.

The paper does not specify the FFT proportionality constant, integer rounding, cross-scale patch alignment, or how predicted codes become embeddings. The choices above document those gaps. The text specifies a last-token context, although Eqs. 14–15 call it `h_cls`; this implementation follows the text. MLP widths, dropout, and optimization settings are recorded in the configuration.

Regression tests cover target-independent predictions, dependence on preceding predicted codes, summed code losses, shared scale parameters, gating, residual quantization, and configuration round trips. Frozen expert targets are cached during stage 3; a regression test checks that cached and direct targets agree.
