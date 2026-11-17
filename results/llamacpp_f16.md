ggml_cuda_init: found 1 CUDA devices (Total VRAM: 8187 MiB):
  Device 0: NVIDIA GeForce RTX 4060 Laptop GPU, compute capability 8.9, VMM: yes, VRAM: 8187 MiB
| model                          |       size |     params | backend    | ngl | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -: | --------------: | -------------------: |
| llama 256M F16                 | 256.63 MiB |   134.52 M | CUDA       |  99 |  0 |     tg64 @ d128 |       359.50 ± 35.89 |
| llama 256M F16                 | 256.63 MiB |   134.52 M | CUDA       |  99 |  0 |     tg64 @ d512 |       363.65 ± 24.94 |
| llama 256M F16                 | 256.63 MiB |   134.52 M | CUDA       |  99 |  0 |    tg64 @ d2048 |        343.39 ± 9.46 |
| llama 256M F16                 | 256.63 MiB |   134.52 M | CUDA       |  99 |  1 |     tg64 @ d128 |       430.60 ± 11.57 |
| llama 256M F16                 | 256.63 MiB |   134.52 M | CUDA       |  99 |  1 |     tg64 @ d512 |       422.53 ± 17.24 |
| llama 256M F16                 | 256.63 MiB |   134.52 M | CUDA       |  99 |  1 |    tg64 @ d2048 |       393.27 ± 17.11 |

build: b7ad48ebd (8661)
