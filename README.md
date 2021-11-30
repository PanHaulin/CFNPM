This repository is focus on Model Compression for Non-parametric Models using deep learning.
## TODO
- [x] memory limitation about kernel metrics or datasets
- [ ] add some large datasets
- [ ] completing gumble-softmax component
- [ ] completing all available methods
- [ ] completing all available workflow
- [ ] add other baselines

## Available Components
- [x] CNN-based compression <font color=#1ABC9C>- Tested</font>
- [ ] Gumble-softmax-based compression
- [x] Class-seperated input <font color=#E74C3C>- Untested</font>
- [x] Instance-added output <font color=#E74C3C>- Untested</font> 

## Available Methods
- [x] Support Vector Classification(SVC) <font color=#1ABC9C>- Tested</font>
- [x] Support Vector Regression(SVR) <font color=#E74C3C>- Untested</font>
- [ ] Kernel Logistic Regression(KLR)
- [ ] Kernel Ridge Regression(KRR)
- [ ] Gaussian Process Regression(GP)
- [ ] K-NearestNeighbor Classification(KNN)
- [ ] K-NearestNeighbor Regression(KNR)

## Available Dataset-based Baselines
- [x] Prototype Generation <font color=#1ABC9C>- Tested</font>
- [x] Prototype Selection <font color=#1ABC9C>- Tested</font>
- [x] Stratified Sampling <font color=#E74C3C>- Untested</font>

## Available Methode-based Basedlines
- [ ] SVC: NSSVM

## Available Workflow
- [x] main <font color=#1ABC9C>- Tested</font>
- [ ] train original model only
- [ ] train baseline only
- [ ] do compression only
- [ ] test compression results only

## Setup: New a conda env and install required packages
```
pip install -r requirements.txt
conda env create -f env.yaml
conda activate pt-rapids
pup install -r requirements.txt
```

## Run Bash files in folder
1. Main 
```bash
# Optional args:
# --patience    EarlyTsopping Callbacks
# --fast_dev_run    Debug Mode
# any other args in args/setup.py and selected components' add_specific_args() function

# SVC
bash bash_files/svc.sh
```

2. Workflow
```bash
Waiting...
```

## Result

Experimental records showed in **Neptune**: https://app.neptune.ai/o/cfnp/-/people

Experimental results showed in **yuque**: https://www.yuque.com/zaisanxuzhongshi/yu74o2

## Files Organization
```
.
├── datasets
├── checkpoints
├── temp
├── bash_files
│   └── svc.sh
│
├── cfnp
│   ├── args
│   │   ├── dataset.py
│   │   └── setup.py
│   ├── baselines
│   │   ├── base.py
│   │   ├── prototype_generation.py
│   │   ├── prototype_selection.py
│   │   └── stratified_sampling.py
│   │   
│   ├── methods
│   │   ├── base.py
│   │   ├── gp.py
│   │   ├── klr.py
│   │   ├── knn.py
│   │   ├── knr.py
│   │   ├── krr.py
│   │   ├── svc.py
│   │   └── svr.py
│   │   
│   ├── modules
│   │   ├── conv.py
│   │   └── gumble.py
│   │   
│   ├── utils
│   │   ├── checkpointer.py
│   │   ├── dm_factory.py
│   │   ├── gen_nssvm_data.py
│   │   ├── helper.py
│   │   ├── km.py
│   │   └── load_data.py
│   │   
│   └── workflows
│       ├── train_basline.py
│       ├── train_np_models.py
│       └── train_ours.py
│
├── main.py
├── test.py
├── requirements.txt
├── env.yaml
└── README.md
```

