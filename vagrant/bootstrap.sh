#!/usr/bin/env bash

echo "Provisioning system. Go have a coffee..."
apt-get update

apt-get install -y linux-headers-$(uname -r)
cd /opt
wget -q http://download.virtualbox.org/virtualbox/5.0.16/VBoxGuestAdditions_5.0.16.iso 
mount VBoxGuestAdditions_5.0.16.iso -o loop /mnt
cd /mnt
sh VBoxLinuxAdditions.run --nox11
cd /opt
rm *.iso
systemctl enable vboxadd
systemctl start vboxadd

echo "Installing NEST dependencies"
apt-get install -y build-essential python3 python3-dev libreadline-dev gsl-bin libgsl0-dev libncurses5-dev
echo "/home/vagrant/opt/nest/lib/python3.4/site-packages/" > /usr/local/lib/python3.4/dist-packages/nest.pth
echo "/home/vagrant/opt/nest/lib/python3.4/site-packages/nest" >> /usr/local/lib/python3.4/dist-packages/nest.pth
echo "Done"

echo "Installing tensorflow"
apt-get install -y python3-pip
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.9.0rc0-cp34-cp34m-linux_x86_64.whl
pip3 install --upgrade $TF_BINARY_URL
echo "Done"

echo "Installing numpy, scipy, matplotlib, scikit-learn"
apt-get install -y gfortran libopenblas-dev libfreetype6-dev 
# For matplotlib
apt-get install -y python3-gi libgtk-3-dev python3-cairocffi
pip3 install numpy scipy matplotlib scikit-learn
echo "Done"

echo "Installing other miscellaneous packages"
apt-get install -y git vim ipython3 python3-tk
echo "Done"

