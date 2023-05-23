# Cross-Modal Global Interaction and Local Alignment for Audio-Visual Speech Recognition

[Cross-Modal Global Interaction and Local Alignment for Audio-Visual Speech Recognition](https://arxiv.org/abs/2305.09212)

## Introduction

This paper proposes a cross-modal global interaction and local alignment (GILA) approach for audio-visual speech recognition (AVSR). The motivation is capture deep audio-visual correlations from both global and local perspectives, which are under-exploited in previous methods.


First, we build a global interaction (GI) model to capture A-V complementary relationship on modality level:

<div align=center>
<img width="825" alt="image" src="https://github.com/YUCHEN005/GILA/assets/90536618/2c3fda49-4028-4318-bfa8-590fdd7ade7b">
</div>

Figure 1: Block diagrams of proposed GILA: (a) Overall architecture, (b) Global Interaction model, (c) Iterative Refinement module.
The $\mathcal{L}\_{ASR}$ denotes speech recognition loss, and $\mathcal{L}\_{LA}$ denotes local alignment loss.

Based on that, we further design a local alignment approach to model A-V temporal consistency on frame level:

<div align=center>
<img width="900" alt="image" src="https://github.com/YUCHEN005/GILA/assets/90536618/edf6058f-8ecc-4ff1-8976-0c4e23bb8987">
</div>

Figure 2: Block diagrams of proposed local alignment approach: (a) Overview, (b) Within-Layer contrastive learning, (c)
Cross-Layer contrastive learning.

Code is coming soon.
