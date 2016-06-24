echo "Installing NEST"
build_dir=$HOME/build
mkdir $build_dir
cd $build_dir
wget -q https://github.com/nest/nest-simulator/releases/download/v2.10.0/nest-2.10.0.tar.gz
tar xf nest-2.10.0.tar.gz
cd nest-2.10.0
PYTHON=/usr/bin/python3 ./configure --prefix=$HOME/opt/nest && make -j2 -s && make install
echo "Done"

echo "Provisioning done"

