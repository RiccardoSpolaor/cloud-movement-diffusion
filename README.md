# Cloud Movement Diffusion
Develop models for accurate cloud movement forecasting and short-term weather predictions to enhance weather related applications through diffusion processes.

Conditioning denoising process on past frames for next-frame prediction:
- Condition the diffusion process on previous frame, to generate future frames
- The model will learn patterns from the past frames and generate an image of the expected movement.
- Condition frames are passed as channels. It has been proved in other cases to be effective in conditioning (e.g. stable diffusion depth and inpainting models).
