# **Dataset**

We use small subset of fastMRI singlecoil knee dataset.
Dataset consist only from PD, 3T scans and slices selected only at center of knee (dataset without slices on knee borders)

[Link to dataset (4 Gb)](https://drive.google.com/file/d/1y78Ad6WwQpMGtxfEZlp97A0iV98kAiJN/view?usp=sharing)

# **Models**

**Unet16**

| noise       | SSIM   | NMSE   | PSNR    |
|-------------|--------|--------|---------|
| [none](https://drive.google.com/file/d/1vBB8AqD_KJAnaf5vJpYNWNq02KWtoW2O/view?usp=sharing)        | 0.8009 | 0.0103 | 31.7120 |
| gaussian    | -      | -      | -       |
| poisson     | -      | -      | -       |
| salt&pepper | -      | -      | -       |

**Cascade-5x-Unet16**

| noise       | SSIM   | NMSE   | PSNR    |
|-------------|--------|--------|---------|
| [none*](https://drive.google.com/file/d/1qTMPzKjURdse352d0bKWPQuh5f2Y53-V/view?usp=sharing)        | 0.8308 | 0.0083 | 32.6528 |
| gaussian    | -      | -      | -       |
| poisson     | -      | -      | -       |
| salt&pepper | -      | -      | -       |

**DnCNN**

| noise       | SSIM   | NMSE   | PSNR    |
|-------------|--------|--------|---------|
| [none](https://drive.google.com/file/d/1vjjsO20eXZ-BwsgHVk23L7NPmATr_COx/view?usp=sharing)        | 0.7742 | 0.0148 | 30.1280 |
| [gaussian](https://drive.google.com/file/d/16h0qD7d5cCVnzkKOCJlttBCdbz2oCTuP/view?usp=sharing)    | 0.6676      | 0.0215      | 28.5129       |
| [salt&pepper](https://drive.google.com/file/d/1paKZwqWPqoRmc3crRtiJ7TQNmQ5F7GqO/view?usp=sharing)     | 0.3955      | 0.0827      | 23.2475       |
| [gaussian + salt&pepper](https://drive.google.com/file/d/1fsARjj3pvoCNbshdPC14OWEqcilkCZf9/view?usp=sharing) | 0.3467      | 0.1060      | 22.3551       |

**Cascade-5x-DnCNN**

| noise       | SSIM   | NMSE   | PSNR    |
|-------------|--------|--------|---------|
| [none*](https://drive.google.com/file/d/1sW2ZOBf26UkViBRBpUDesHPxlILUaoiS/view?usp=sharing)        | 0.8341 | 0.0079 | 32.9552 |
| gaussian    | -      | -      | -       |
| poisson     | -      | -      | -       |
| salt&pepper | -      | -      | -       |
    * - need revision