# ray_simulator 
evaluation of ip

## Installation 
1. Install anaconda3.7
2. Install ceres http://ceres-solver.org/installation.html
3. Install conda packages: pybind11, numpy, matplotlib, shapely, tqdm
4. Install via pip in conda env: svgpathtools
5. Install Tina’s Random Number Generator Library Version 4
```sh
    download latest version https://www.numbercrunch.de/trng/
    extract files
    cd trng-4.* 
    ./configure
    make
    sudo make install
```
## Build native code 
 open conda env in terminal
 ```sh
    cd ${Project_Name}/native 
    mkdir build 
    cd build 
    cmake ..
    make install
 ```
### run code
in python console 
```sh
   execfile('scan_match_filter_evaluation.py')
 ```
execfile('scan_match_filter_evaluation.py')
