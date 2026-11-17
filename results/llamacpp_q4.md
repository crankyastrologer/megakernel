ggml_cuda_init: found 1 CUDA devices (Total VRAM: 8187 MiB):
  Device 0: NVIDIA GeForce RTX 4060 Laptop GPU, compute capability 8.9, VMM: yes, VRAM: 8187 MiB
| model                          |       size |     params | backend    | ngl | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -: | --------------: | -------------------: |
| llama 256M Q4_K - Medium       |  98.87 MiB |   134.52 M | CUDA       |  99 |  0 |     tg64 @ d128 |       427.60 ± 58.79 |
| llama 256M Q4_K - Medium       |  98.87 MiB |   134.52 M | CUDA       |  99 |  0 |     tg64 @ d512 |       490.42 ± 24.31 |
| llama 256M Q4_K - Medium       |  98.87 MiB |   134.52 M | CUDA       |  99 |  0 |    tg64 @ d2048 |       467.64 ± 23.55 |
| llama 256M Q4_K - Medium       |  98.87 MiB |   134.52 M | CUDA       |  99 |  1 |     tg64 @ d128 |       571.52 ± 37.68 |
| llama 256M Q4_K - Medium       |  98.87 MiB |   134.52 M | CUDA       |  99 |  1 |     tg64 @ d512 |       588.69 ± 20.41 |
| llama 256M Q4_K - Medium       |  98.87 MiB |   134.52 M | CUDA       |  99 |  1 |    tg64 @ d2048 |       530.84 ± 19.27 |

build: b7ad48ebd (8661)
