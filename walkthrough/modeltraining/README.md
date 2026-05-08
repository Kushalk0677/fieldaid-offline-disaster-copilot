# `modeltraining/`

Dataset download and staging utilities.

This folder intentionally does not include the full QCRI/MEDIC image dataset or Hugging Face cache because those files are very large. The public repo keeps only the downloader and documentation needed to reproduce the dataset preparation.

Use `modeltraining/medic_qcri/download_medic.py` to rebuild the local dataset subset when needed.
