# **Dataset**

We use small subset of fastMRI singlecoil knee dataset.
Dataset consist only from PD, 3T scans and slices selected only at center of knee (dataset without slices on knee borders)

[Link to dataset (4 Gb)](https://drive.google.com/file/d/1y78Ad6WwQpMGtxfEZlp97A0iV98kAiJN/view?usp=sharing)
You should have h5py > 3.2 and gdown > 3.12, you can update like that:

`python -m pip install gdown==3.12.2`

`python -m pip install h5py==3.2.1`

# **Models**

**Unet16**

| noise       | SSIM   | NMSE   | PSNR    |
|-------------|--------|--------|---------|
| [none](https://drive.google.com/file/d/1vBB8AqD_KJAnaf5vJpYNWNq02KWtoW2O/view?usp=sharing)        | 0.8009 | 0.0103 | 31.7120 |
| gaussian    | -      | -      | -       |
| poisson     | -      | -      | -       |
| salt&pepper | -      | -      | -       |

**Cascade-5x-Unet16-DC**

| noise       | SSIM   | NMSE   | PSNR    |
|-------------|--------|--------|---------|
| [none](https://drive.google.com/file/d/1qTMPzKjURdse352d0bKWPQuh5f2Y53-V/view?usp=sharing)        | 0.8444 | 0.0069 | 33.4667 |
| [gaussian](https://drive.google.com/file/d/16LiGoQwz0HdtJ2x084Xrld6lqQIMxqxc/view?usp=sharing)    | 0.6035 | 0.0242 | 28.3150 |
| poisson     | -      | -      | -       |
| [salt&pepper](https://drive.google.com/file/d/13HttRoGv_Oh7lpB0qp7HLI8ZL4rDqqoR/view?usp=sharing) | 0.6156 | 0.0262 | 28.2839 |

**DnCNN**

| noise       | SSIM   | NMSE   | PSNR    |
|-------------|--------|--------|---------|
| [none](https://drive.google.com/file/d/1vjjsO20eXZ-BwsgHVk23L7NPmATr_COx/view?usp=sharing)        | 0.7742 | 0.0148 | 30.1280 |
| [gaussian](https://drive.google.com/file/d/16h0qD7d5cCVnzkKOCJlttBCdbz2oCTuP/view?usp=sharing)    | 0.6676      | 0.0215      | 28.5129       |
| [salt&pepper](https://drive.google.com/file/d/1paKZwqWPqoRmc3crRtiJ7TQNmQ5F7GqO/view?usp=sharing)     | 0.3955      | 0.0827      | 23.2475       |
| [gaussian + salt&pepper](https://drive.google.com/file/d/1fsARjj3pvoCNbshdPC14OWEqcilkCZf9/view?usp=sharing) | 0.3467      | 0.1060      | 22.3551       |

**Cascade-5x-DnCNN-DC**

| noise       | SSIM   | NMSE   | PSNR    |
|-------------|--------|--------|---------|
| [none](https://drive.google.com/file/d/1sW2ZOBf26UkViBRBpUDesHPxlILUaoiS/view?usp=sharing)        | 0.8394 | 0.0072 | 33.2991 |
| [gaussian](https://drive.google.com/file/d/1df7xelNU7QNY9tuqoMjUeePCdpZbW0S1/view?usp=sharing)    | 0.6639      | 0.0182      | 29.5485       |
| [salt&pepper](https://drive.google.com/file/d/1BYWryHtXWSkRlP1l-frp6z_hRJU-DQjY/view?usp=sharing)     | 0.3112      | 0.2565      | 19.3469       |
| [gaussian + salt&pepper](https://drive.google.com/file/d/1HRlUVJXR-ps6Cz-t355ODa0FNbqxYxNy/view?usp=sharing) | 0.2022      | 0.5462      | 16.1158       |

**Cascade-5x-DnCNN-DCL**

| noise       | SSIM   | NMSE   | PSNR    |
|-------------|--------|--------|---------|
| [none](https://drive.google.com/file/d/1P0cOfqj4CeqtGTZyj0gg_zUi0gEdokTo/view?usp=sharing)        | 0.8325 | 0.0078 | 32.9421 |
| [gaussian](https://drive.google.com/file/d/1bkfvY6573ZWt752kFiy4NhD6P_B0sUgX/view?usp=sharing)    | 0.7098      | 0.0138      | 30.4711       |
| [salt&pepper](https://drive.google.com/file/d/1jzMuxcEW2tOgrQ8U8VBVR8nOe8GCUJtq/view?usp=sharing)     | 0.4911      | 0.0420      | 25.9640       |
| [gaussian + salt&pepper](https://drive.google.com/file/d/1TuPPuht1OwfwJ_9xuikWOjHAY_xt9R8y/view?usp=sharing) | 0.4276      | 0.0693      | 24.4179       |

    * - need revision
