# Third-Party Notices

`comfyui-ne-tools` includes vendored or adapted code from the following open-source projects. Model weights under `model/` are not distributed with this repository and remain subject to their respective upstream licenses.

---

## 1. BiRefNet (`py/core/rembg/`)

- **Upstream:** <https://github.com/ZhengPeng7/BiRefNet>
- **Copyright:** Copyright (c) 2024 Peng Zheng
- **License:** MIT License

> **Note on model weights:** `model/rembg/birefnet.safetensors` is supplied by the user. Official `ZhengPeng7/BiRefNet` checkpoints are released under the MIT License; `briaai/RMBG-2.0` weights built on the BiRefNet architecture are released under a separate non-commercial license (CC BY-NC 4.0). Check the license of whichever checkpoint file you place in `model/rembg/`.

---

## 2. ComfyUI-WanVideoWrapper (`py/core/wan/wan_nodes_utility.py`, `py/core/wan/utils.py`)

- **Upstream:** <https://github.com/kijai/ComfyUI-WanVideoWrapper>
- **Copyright:** Copyright (c) 2025 Jukka Seppänen (kijai) and contributors
- **License:** Apache License, Version 2.0 (<https://www.apache.org/licenses/LICENSE-2.0>)

Licensed under the Apache License, Version 2.0 (the "License"); you may not use these files except in compliance with the License. Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

---

## 3. SDXL Latent Resizer (`py/core/latent_resizer.py`)

- **Upstream:** <https://huggingface.co/Birchlabs/sdxl-resizer>
- **Copyright:** Copyright (c) 2023 Alex Birch (Birchlabs)
- **License:** MIT License

---

## 4. ComfyUI-Custom-Scripts & textarea-caret-position (`js/autocomplete.js`, `js/autocomplete.css`)

- **Upstream:**
  - <https://github.com/pythongosssss/ComfyUI-Custom-Scripts> — Copyright (c) pythongosssss
  - <https://github.com/component/textarea-caret-position> — Copyright (c) 2015 Jonathan Ong, Vishal Kurup
- **License:** MIT License

---

## 5. rgthree-comfy (`js/muter.js`, `js/services/fast_groups_service.js`, `js/draw.js`, `js/base.js`)

- **Upstream:** <https://github.com/rgthree/rgthree-comfy>
- **Copyright:** Copyright (c) Regis Gaughan (rgthree)
- **License:** MIT License
