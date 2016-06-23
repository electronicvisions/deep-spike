# Deep Spike

## Installation guide
This section describes how to set up your computer to be able to run the tensorflow and NEST code in this repository.

You have two options:

1. Install all packages manually on your system (preferred for long running simulations, use on clusters, general high
   performance requirements)
1. Use the vagrant setup instructions in the vagrant directory to let the scripts automatically setup the development
   environment for you (use on Windows, possibly Mac OS X; for development and testing)

### Manual installation steps (for Debian/Ubuntu systems, x86_64, Python 3)
#### Dependencies
To install all the required dependencies, run:

    # For NEST
    sudo apt-get install -y build-essential python3 python3-dev libreadline-dev gsl-bin libgsl0-dev libncurses5-dev
    # For tensorflow
    sudo apt-get install -y python3-pip
    # For numpy, scipy, scikit-learn
    sudo apt-get install -y gfortran libopenblas-dev libfreetype6-dev 
    # For matplotlib
    sudo apt-get install -y python3-gi libgtk-3-dev python3-cairocffi
    # Misc packages
    sudo apt-get install -y git vim ipython3 python3-tk

#### Python libraries
To install required Python libraries, run:

    sudo pip3 install numpy scipy matplotlib scikit-learn

While you can install these packages using apt-get, the pip install gives you newer versions. Also, numpy and scipy are
automatically compiled with OpenBLAS with the pip install (if you don't have any other BLAS package installed), which is extremely performant.

#### Tensorflow
To install Tensorflow, run:

    export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.9.0rc0-cp34-cp34m-linux_x86_64.whl
    sudo pip3 install --upgrade $TF_BINARY_URL

See the [Tensorflow documentation](https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html#pip-installation) for more options and details

#### NEST
To install NEST, run:

    build_dir=$HOME/build
    mkdir $build_dir
    cd $build_dir
    # Download NEST
    wget -q https://github.com/nest/nest-simulator/releases/download/v2.10.0/nest-2.10.0.tar.gz
    tar xf nest-2.10.0.tar.gz
    cd nest-2.10.0
    # Build and install NEST
    PYTHON=/usr/bin/python3 ./configure --prefix=$HOME/opt/nest && make -j2 -s && make install

Then add the NEST python packages to your PYTHONPATH by adding the following line to your .zshenv/.bash_profile/.profile: 

    export PYTHONPATH=$HOME/opt/nest/lib/python3.4/site-packages/:$HOME/opt/nest/lib/python3.4/site-packages/nest:$PYTHONPATH

Alternatively, you can add it to the sytem path with the following (although this is not the nicest thing to do):

    sudo echo "$HOME/opt/nest/lib/python3.4/site-packages/" > /usr/local/lib/python3.4/dist-packages/nest.pth
    sudo echo "$HOME/opt/nest/lib/python3.4/site-packages/nest" >> /usr/local/lib/python3.4/dist-packages/nest.pth

See the [Nest documentation](http://www.nest-simulator.org/installation/) for details and more options.

#### Notes
* You can switch python3 with python2 (and pip3 with pip) in the above installation steps if you prefer Python 2 (If you do this, see
  [this](http://i.imgur.com/wqvo221.gif)) 
* If you want to install everything locally without sudo, you can replace all the ```pip3 install``` steps with ```pip3 install --user```.
  NEST is already installed locally. Obviously, you would still need sudo for installing the dependencies with apt-get.

## Running the code
Run ```run.sh``` from the 'deep-spike' directory. This script first builds the C++ tensorflow ops, and then runs the
python experiment.

NOTE: If you're running the code from pycharm (or some other IDE), be sure to set the working directory to the root directory
(i.e. deep-spike)

## Directory structure
```
deep-spike
|-bin : Contains runnable python experiments
|-cc : contains C++ tensor operations, python gradient implementations for these operations, and local tests
|-lib : ??
|-vagrant : Contains files for vagrant setup
```