# Data (not committed)

The datasets are licensed for non-commercial research and are never committed to git.

```
data/
├── WFLW/                          # required for training / evaluation
│   ├── WFLW_images/               # 6,551 photos in event folders (from WFLW_images.tar.gz)
│   └── WFLW_annotations/
│       ├── list_98pt_rect_attr_train_test/
│       │   ├── list_98pt_rect_attr_train.txt   # 7,500 faces
│       │   └── list_98pt_rect_attr_test.txt    # 2,500 faces
│       └── list_98pt_test/                     # official per-condition test lists
├── cache/                         # written by `python -m src.dataset` (pre-cropped faces)
└── APAC/                          # optional: folder or .zip of {id}_{Group}.jpg face images
```

Download WFLW from the links in https://github.com/wywu/LAB (README, "WFLW Dataset Download"),
then extract both archives here:

```bash
tar -xzf WFLW_images.tar.gz -C data/WFLW
tar -xzf WFLW_annotations.tar.gz -C data/WFLW
```

The official annotation link on the author's site has gone offline; an archived copy is
available from the Internet Archive (web.archive.org) under the same URL.
