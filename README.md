# **Dataset**

We use small subset of fastMRI singlecoil knee dataset.
Dataset consist only from PD, 3T scans and slices selected only at center of knee (dataset without slices on knee borders)

[Link to dataset (4 Gb)](https://drive.google.com/file/d/1y78Ad6WwQpMGtxfEZlp97A0iV98kAiJN/view?usp=sharing)
You should have h5py > 3.2 and gdown > 3.12, you can update like that:

`python -m pip install gdown==3.12.2`

`python -m pip install h5py==3.2.1`

# **Noise transform**

**Gaussian**
```
from k_space_reconstruction.datasets.fastmri import FastMRITransform, RandomMaskFunc

transform = FastMRITransform(
    RandomMaskFunc([0.08], [4]),
    noise_level=100,
    noise_type='normal'
)
```

**Salt**
```
from k_space_reconstruction.datasets.fastmri import FastMRITransform, RandomMaskFunc

transform = FastMRITransform(
    RandomMaskFunc([0.08], [4]),
    noise_level=5e4,
    noise_type='salt'
)
```

**Gaussian + Salt**
```
from k_space_reconstruction.datasets.fastmri import FastMRITransform, RandomMaskFunc

transform = FastMRITransform(
    RandomMaskFunc([0.08], [4]),
    noise_type='normal_and_salt'
)
```

# **Models**

**Unet16**

| noise       | SSIM   | NMSE   | PSNR    |
|-------------|--------|--------|---------|
| [none](https://drive.google.com/file/d/113DZqjgACZHdRxC3xRERD5hOMZtaRYI_/view?usp=sharing)        | 0.8053 | 0.0099 | 31.8321 |
| [gaussian](https://drive.google.com/file/d/1S9TMhP2g8UOjOpXggO4dPLGq5FLl84S2/view?usp=sharing)    | 0.7210 | 0.0142 | 30.3041 |
| [salt&pepper](https://drive.google.com/file/d/1DhFYzpAnX25jQwMe78l_P17yfvcWdXJx/view?usp=sharing)     | 0.6806 | 0.0207 | 28.9547 |
| [gaussian + salt&pepper](https://drive.google.com/file/d/1puD_V3z87IXsFCqQiiNroeFrI5x1owlI/view?usp=sharing) | 0.6807 | 0.0189 | 28.2086 |

**Attention-Unet16**

| noise       | SSIM   | NMSE   | PSNR    |
|-------------|--------|--------|---------|
| [none](https://drive.google.com/file/d/1pOzpSr4T-VBF50puf0fbdC_0FQSJDSrZ/view?usp=sharing)        | 0.4939 | 0.2021 | 18.6097 |
| [gaussian](https://drive.google.com/file/d/1erIiqUxt5yzkm89552FO-cePM5cuXdty/view?usp=sharing)    | 0.4426 | 0.2127 | 18.3985 |
| [salt&pepper](https://drive.google.com/file/d/1jivdYTQ67z2JPffkJOh2KRE7r7YNhcXH/view?usp=sharing)     | 0.3875 | 0.3336 | 16.6910 |
| [gaussian + salt&pepper](https://drive.google.com/file/d/1LRHMG03gNwK00dgOkkW_GpeeO3X9OkvQ/view?usp=sharing) | 0.3624 | 0.3872 | 16.1630 |

**Cascade-5x-Unet16-noDC**

| noise       | SSIM   | NMSE   | PSNR    |
|-------------|--------|--------|---------|
| [none](https://drive.google.com/file/d/1AqLmq6SaPaKAfNlGKEdPvJezkU3Kuevx/view?usp=sharing)        | 0.8013 | 0.0097 | 31.9888 |
| [gaussian](https://drive.google.com/file/d/1dXDukIDxyvq1iwCX6xIwx4IzHKSaHD0v/view?usp=sharing) | 0.7058 | 0.0151 | 30.0426 |
| [salt&pepper](https://drive.google.com/file/d/1iLYBxanMpz9wghbt5zlI_6pz0XA_cak6/view?usp=sharing) | 0.6169 | 0.0394 | 26.5311 |
| [gaussian + salt&pepper](https://drive.google.com/file/d/1cme5Fu4kfNa9ZywifJ-VpI28q9gwEWIE/view?usp=sharing)  | 0.6086 | 0.0574 | 25.8732|

**Cascade-5x-Unet16-DCL**

| noise       | SSIM   | NMSE   | PSNR    |
|-------------|--------|--------|---------|
| [none](https://drive.google.com/file/d/1qTMPzKjURdse352d0bKWPQuh5f2Y53-V/view?usp=sharing)        | 0.8444 | 0.0069 | 33.4667 |
| [gaussian](https://drive.google.com/file/d/13dvvJA4K00mr9xXhxpT82vhuvjxnjIF1/view?usp=sharing) | 0.7388 | 0.0117 | 31.2349 |
| [salt&pepper](https://drive.google.com/file/d/13HttRoGv_Oh7lpB0qp7HLI8ZL4rDqqoR/view?usp=sharing) | 0.6156 | 0.0262 | 28.2839 |
| [gaussian + salt&pepper](https://drive.google.com/file/d/1472YeD5lQcbB8fbn5cGP3lXgGB3Maldt/view?usp=sharing)  | 0.5892 | 0.03026 | 27.8198|
| [gaussian_400](https://drive.google.com/file/d/16LiGoQwz0HdtJ2x084Xrld6lqQIMxqxc/view?usp=sharing)    | 0.6035 | 0.0242 | 28.3150 |
| [gaussian_400 + salt&pepper](https://drive.google.com/file/d/1BLTuQywe0lJI6cLfU_35iOEQ131Nzv60/view?usp=sharing)     | 0.5262 | 0.0419 | 26.2551 |

**Cascade-5x-Unet16-DC**

| noise       | SSIM   | NMSE   | PSNR    |
|-------------|--------|--------|---------|
| [none](https://drive.google.com/file/d/13FaJ5itN6OkYoxY_qGy4eyZFwI0YnuL5/view?usp=sharing) | 0.8508 | 0.0064 | 33.8926 |
| [gaussian](https://drive.google.com/file/d/13oQoujDBUKqoMfVRb-vV0eSsiv5gFGQP/view?usp=sharing) | 0.6747 | 0.0171 | 29.8601|
| [salt&pepper](https://drive.google.com/file/d/1D8kk67tjO2lbBPv6xprYTJLqsHi1BlyA/view?usp=sharing)  | 0.3862 | 0.1945 | 21.1475|
| [gaussian + salt&pepper](https://drive.google.com/file/d/14w64gbtbXxab3ad8kMY-tkR8CoEp26ZL/view?usp=sharing) | 0.2580 | 0.3408 | 18.0605|

**Cascade-5x-Unet16-DC-AF**

| noise       | SSIM   | NMSE   | PSNR    |
|-------------|--------|--------|---------|
| [none](https://drive.google.com/file/d/1OWx0XrJ_fjyaiy9JHJfEUtUkSuFL9CgU/view?usp=sharing)                      | 0.8419 | 0.0071 | 33.3439 |
| [gaussian](https://drive.google.com/file/d/1FaGhK7IjxPGUhxgkK_9Zo6xH9AbR8cG1/view?usp=sharing)                  | 0.7254 | 0.0125 | 30.9641 |
| [salt&pepper](https://drive.google.com/file/d/1FaGhK7IjxPGUhxgkK_9Zo6xH9AbR8cG1/view?usp=sharing)               | 0.5208 | 0.0584 | 25.8676 |
| [gaussian + salt&pepper](https://drive.google.com/file/d/1W7cB_OFJz4TkeEP_l07CCiCRf6HKs_Du/view?usp=sharing)    | 0.5130 | 0.0517 | 25.8669 |

**Cascade-5x-Unet16-DC-Super-AF**

| noise       | SSIM   | NMSE   | PSNR    |
|-------------|--------|--------|---------|
| [none](https://drive.google.com/file/d/12Cit-99AY04anWJ-6mFYqih6XoZ9CcyG/view?usp=sharing)                      | 0.8428 | 0.0070 | 33.4662 |
| [gaussian](https://drive.google.com/file/d/1s2vEEkYFaZrS1yRmOQyuDa_J82s6schm/view?usp=sharing)                  | 0.7508 | 0.0111 | 31.4516 |
| [salt&pepper](https://drive.google.com/file/d/1LDYte3YX8U2krChCMO9F4QtUiUhO6d2z/view?usp=sharing)               | 0.8347 | 0.0082 | 32.8732 |
| [gaussian + salt&pepper](https://drive.google.com/file/d/12ZJL3v_GF7GHRMk6AH5Vye2J0XZMkPki/view?usp=sharing)    | 0.7359 | 0.0133 | 30.7945 |

**Cascade-5x-Unet16-TDC-FtF-Super-AF**

| noise       | SSIM   | NMSE   | PSNR    |
|-------------|--------|--------|---------|
| [none](https://drive.google.com/file/d/1zLIq5bgpOF6zRgqQpVQkGVF-RESif_i3/view?usp=sharing)                      | 0.8509 | 0.0063 | 33.9353 |
| [gaussian](https://drive.google.com/file/d/107VN5R7Rp8SKlVRmXpTjeReN_7N3YWEf/view?usp=sharing)                  | 0.7510 | 0.0109 | 31.5610 |
| [salt&pepper](https://drive.google.com/file/d/1oHgDpLLt53PuWLI3qnScAt7M3X0roMpw/view?usp=sharing)               | 0.8437 | 0.0071 | 33.4054 |
| [gaussian + salt&pepper](https://drive.google.com/file/d/1aJqQl8hSD0rZtZySOuGoO8aBuMJmZZ7y/view?usp=sharing)    | 0.7361 | 0.0121 | 31.1309 |

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


**Cascade-5x-DnCNN-NoDC**

| noise       | SSIM   | NMSE   | PSNR    |
|-------------|--------|--------|---------|
| [none](https://drive.google.com/file/d/1iUbaZcwKAAW7IE26070Yj4v6WJc5fEKr/view?usp=sharing)        | 0.7712 | 0.0154 | 29.9269 |
| [gaussian](https://drive.google.com/file/d/1_le3Kd2jMBFXkuTC2AGUfiwtqyQJiqTU/view?usp=sharing)    | 0.6601      | 0.0227      | 28.3064       |
| [salt&pepper](https://drive.google.com/file/d/1MZM0I9Njqq6embB8x4bHEZ27Y2ikpXPO/view?usp=sharing)     | 0.4276      | 0.0654      | 23.9284       |
| [gaussian + salt&pepper](https://drive.google.com/file/d/1MBrILGhitdZQzmDk0gbES7HvaEnb7baL/view?usp=sharing) | 0.4551      | 0.0717      | 23.9338       |

    * - need revision
