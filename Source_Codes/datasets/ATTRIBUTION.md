# Dataset Sources and Scope of Use

**T-LESS:** Hodaň et al., *T-LESS: An RGB-D Dataset for 6D Pose Estimation of Texture-less Objects*, WACV 2017. [Author-maintained dataset page](https://cmp.felk.cvut.cz/t-less/). License: CC BY 4.0. The experiments use the manually created CAD models in `models_cad`. The original archive also contains reconstructed and evaluation models; these do not replace the CAD inputs used in this study.

**MVTec ITODD:** Drost et al., *Introducing MVTec ITODD – A Dataset for 3D Object Recognition in Industry*, ICCV Workshops 2017. [Author-maintained dataset page](https://www.mvtec.com/research-teaching/datasets/mvtec-itodd). Section 3 of the paper states that manually created CAD models are provided for all 28 objects. License: CC BY-NC-SA 4.0. This study uses the dataset for non-commercial academic research.

Downloads originate from the official BOP dataset repositories on Hugging Face. Versions and SHA-256 hashes are recorded in [dataset_sources.json](../configs/dataset_sources.json). Original models are preserved. Derived assets apply only millimetre-to-metre conversion and deterministic surface sampling. Subsequent support-frame transformations, collision approximations, rendering materials, and simulated physical properties are recorded separately.

CAD geometry, the datasets' original real photographs, and this project's synthetic stereo images are distinct data sources. Mass, friction, and material appearance assigned in simulation are experimental settings, rather than measured properties supplied by the source datasets.
