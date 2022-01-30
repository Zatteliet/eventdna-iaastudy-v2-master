# eventdna-iaastudy-v2

This repo describes the IAA study performed on EventDNA data in late 2019/early 2020.

`main.py` is the entry point to this script. Run-specific parameters can be adjusted there. `main.py` calls functions defined in the various modules in `iaastudy`. Each module is internally documented.

Example run:

```bash
python main.py data/iaa_set_dnaf.zip /mnt/c/Users/camie/Downloads/IAA_OUT/ fallback --restricted
```
