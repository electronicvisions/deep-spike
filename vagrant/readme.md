# Installing using Vagrant

Vagrant creates a lightweight virtual machine based development environment within which all packages can be installed.
The advantage of this is that this is completely platform agnostic. You can run the code from Linux/Mac OS X/Windows
without worrying about cross-platform compatibility, since the virtual machine guest runs linux anyway.

This method is good to get started on your laptop/desktop but not to run long running simulations.

Terminology-wise, your laptop/desktop OS is the 'host' OS (which may be Linux/Mac OS X/Windows) and the virtual machine running using vagrant is the 'guest' OS
or 'guest' VM (which in this case is Debian Jessie).

## Installation
1. Install VirtualBox from https://www.virtualbox.org/wiki/Downloads
1. Install Vagrant from https://www.vagrantup.com/downloads.html
1. Go to the 'deep-spike/vagrant' directory (the directory in which this readme lives) and run ```vagrant up```
1. Go drink a coffee. The setup takes about 15-20 minutes on my Core i5 MacBookPro. The script sets up NEST, Tensorflow,
   python numpy, scipy, scikit-learn etc.  and all required dependencies.
1. Once this is done you can access the newly created virtual machine through the commandline using ```vagrant ssh```
1. Run ```vagrant rsync-auto``` to setup directory sharing between the host and guest OSs. All files in the same
   directory as the file 'Vagrantfile' will be synced one-way into the '/vagrant' directory in the guest OS. NOTE: The
   syncing is one-way only i.e. changes you make in the host OS will be reflected in the guest OS but not vice versa. So
   don't make changes to files from within the guest VM.

## Notes
* The virtualbox guest is debian jessie based.
* All the installation uses python3 (including for PyNEST setup)

## Using PyCharm as an IDE
If you are using PyCharm to write your python code, you can setup PyCharm to pick up all the python libraries from
within the vagrant VM so that it finds all necessary libraries when you write your code from outside the VM.
To do this, setup a 'remote' interpreter for your project, choose the 'Vagrant' option, and choose the directory in
which your 'Vagrantfile' resides. Then change the python path to '/usr/bin/python3' (to make sure it uses Python 3).
The initial sync takes a bit of time.
