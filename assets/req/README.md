# Dependancy management

Use following files to recreate this env

Files description:
```
├── README.md                                               # Readme and instruction
├── env                                                     # env .yml files
│   ├── learn-env-win.yml                                   # for use with windows os
│   └── learn-env.yml                                       # os agnostic
└── package_and_lib                                         # dependancies of this analysis
    ├── requirements_conda.txt                              # use with conda
    └── requirements_pip.txt                                # use with pip
```
NOTE: text inside <> refers to respective names
___

# Installing requirements

If using pip:
```python
pip install -r <requirements>.txt
```
If using conda:
```python
conda create --name <env_name> --file <requirements>.txt
```
NOTE: Creating requirement file:
```python
# using pip
pip freeze > <requirements_pip>.txt
# using conda
conda list --export > <requirements_conda>.txt
```
___

# Enviournment management using Anaconda
## Install saved env
create new env from .yml
```python
conda env create -f <env_name>.yml
```
NOTE: 

create .yml
```python
# show conda list
conda env list
# activate env first
conda activate <env>
conda env export > <env>.yml --no-builds 
```
OR
```python
conda env export > learn-env-new.yml
```
Useful command:
```python
# Create a new environment named py35
conda create --name py35
# Activate the new environment to use it
## WINDOWS: 
conda activate py35
## UNIX-like & macOS: 
source activate py35
```

## Add kernel
List all kernels avaiable
```python
jupyter kernelspec list
```

Add kernel
```python
python -m ipykernel install --user --name <env_name> --display-name "<kernel_name>"
```

NOTE:

Remove kernel
```python
jupyter kernelspec remove <kernel_name>
```