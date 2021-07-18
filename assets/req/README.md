# Dependancy management

Use following files to recreate env used for this analysis.

Files description:
```
├── README.md                                               # Readme and instructions
├── env_files                                               # env .yml files
│   ├── learn-env-win.yml                                   # for use with windows os
│   └── learn-env.yml                                       # os agnostic
└── package_and_lib                                         # dependancies of this analysis
    ├── requirements_conda.txt                              # use with conda
    └── requirements_pip.txt                                # use with pip
```
NOTE: 

[1] Text inside <> refers to respective names

[2] cd to this folder can be helpful

[3] This is tested on a Windows 10 Pro machine using WSL2 (Windwos Subsystem for Linux version 2) terminal of Ubuntu-20.04 Virtual Machine, git-bash version 2.30.0.windows.1, and Microsoft Visual Studio Code version 1.58.2.
___
___

# Steps:
- get a local copy of this repo via git, options are:
    - clone this repo locally
    - fork this repo locally 
    - get zipped version of repo and extract locally
- open terminal in the appropriate folder, env_files OR package_and_lib
- If using a fresh conda env:
    - open terminal in ```env_files``` folder
    - use appropriate .yml. e.g., ```learn-env.yml```
    
        ```conda env create -f learn-env.yml```
    - add kernel: replace ```<text>s```
        - ```python -m ipykernel install --user --name <env_name> --display-name "<kernel_name>" ```
    - use the kernel along the new env

- If installing required packages in the existing env
    - open terminal in ```package_and_lib``` folder
    - using conda: <env_name> == new env name 
    
        ```conda create --name <env_name> --file requirements_conda.txt```
    - using pip:

        ```pip install -r requirements_pip.txt```

<br>

___
___

# Some Useful commands
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
conda env export > <env>.yml --no-builds # os agnostic
```
OR
```python
conda env export > <env>.yml # os specific
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

___
___

DISCLAIMER: 

Author does not take any responsibilty if you manage to break your env setup. But if that bad luck ever bestow upon you, dont feel hesitant to contact.