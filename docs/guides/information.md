# Information guide concerning all the technical details of the project

---

## Directory Tree Structure

The following project operates under the following directory tree structure for storing the data:

```cmd
data
├── classification
│   ├── train
│   │   ├── cancer
│   │   └── no_cancer
│   ├── valid
│   │   ├── cancer
│   │   └── no_cancer
│   └── test
│       ├── cancer
│       └── no_cancer
├── segmentation
│   ├── train
│   │   ├── images
│   │   └── masks
│   ├── valid
│   │   ├── images
│   │   └── masks
│   └── test
│       ├── images
│       └── masks
└── detection
    ├── train
    │   ├── images
    │   └── annotations
    ├── valid
    │   ├── images
    │   └── annotations
    └── test
        ├── images
        └── annotations
```

This directory tree is built upon a given ```dataset``` directory.

---

## Automatic Documentation

Documentation of the project is automatically generated using ```sphinx```.

**Docstrings** are written in REST (reStructuredText) format (check point **2** in the **Bibliography** section for more information on how to write REST).

---

## Dependencies

Dependencies are dumped in the ```requirements.txt``` file. If a new dependency is added, the whole file must be updated like this:

```cmd
pip freeze > requirements.txt
```

A *virtual enviroment* **must** be used for this in order to prevent incompatibility between dependencies.

---

## Bibliography

1. [(temporal, borrar cuando encontremos algo mejor) Dataset](https://universe.roboflow.com/upm-alyry/breast-cancer-bounding-box)

2. [Quick reStructuredText](https://docutils.sourceforge.io/docs/user/rst/quickref.html) 

3. [Conventional Commit Guide](https://www.conventionalcommits.org/en/v1.0.0/#summary)

4. [ResNet Paper](https://arxiv.org/abs/1512.03385)