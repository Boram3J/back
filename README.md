# Installation

Requires `python>=3.9` and `cuda==1.16`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### OCR Demo

Download weights to `ocr/weights` as described in WORD-Pytorch

```bash
python ocr/demo.py --ocr
```

## Reference

- https://fastapi.tiangolo.com/tutorial/bigger-applications
