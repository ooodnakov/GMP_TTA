# GMP_TTA
Enhancing Semantic Segmentation of LiDAR Point Clouds through Global Maps

This is a code repository of my Thesis work.

You can find all `.py` files and notebooks here.


## How to
I hade following workdir structure:
```
.
└── workdir/
    ├── dataset/
    │   └── sequences/
    │       └── ...
    ├── gits/
    │   └── ...
    ├── utils/
    │   ├── one.py
    │   └── two.py
    ├── .py files
    └── .ipynb files
    └── Pipfile
```

Then I created pipenv env and got provided Pipfile .

```bash
pip install --user pipenv
pipenv --python 3.10
pipenv install
```

Files in `utils` folder - are modified files from [Semantic Kitti API repo](https://github.com/PRBonn/semantic-kitti-api).

Then you can created parts of map with following command:
```bash
pipenv run python map_creators.py 08 0 1000
pipenv run python map_creators.py 08 500 1500
...
```

And with depth-weighted samplaing, one can crate GMP:
```bash
pipenv run python velodyne2_creator.py 08 0 1000 2
pipenv run python velodyne2_creator.py 08 500 1500 2
...
```

And with uniform:
And with depth-weighted samplaing, one can crate GMP:
```bash
pipenv run python velodyne2_creator.py 08 0 1000 3
pipenv run python velodyne2_creator.py 08 500 1500 3
...
```


Notebooks were primarly used as playgrounds.

And with `visualize.py` script one can view open3d windows with point clouds.