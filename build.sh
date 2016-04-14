#!/bin/bash

mkdir deps &> /dev/null
cd deps

#Add necessary extra repos
version=$(lsb_release -a 2>&1)
if [[ $version == *"14.04"* ]] ; then
    wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.5-18_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu1404_7.5-18_amd64.deb
    rm cuda-repo-ubuntu1404_7.5-18_amd64.deb
    sudo add-apt-repository -y ppa:v-launchpad-jochen-sprickerhof-de/pcl
    sudo apt-get update
    sudo apt-get install -y libpcl-all
elif [[ $version == *"15.04"* ]] ; then
    wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1504/x86_64/cuda-repo-ubuntu1504_7.5-18_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu1504_7.5-18_amd64.deb
    rm cuda-repo-ubuntu1504_7.5-18_amd64.deb
    sudo apt-get update
    sudo apt-get install -y libpcl-dev yasm libvtk5-qt4-dev
else
    echo "Don't use this on anything except 14.04 or 15.04"
    exit
fi

sudo apt-get install -y cmake-qt-gui git build-essential libusb-1.0-0-dev libudev-dev openjdk-7-jdk freeglut3-dev python-vtk libvtk-java libglew-dev cuda-7-5 libsuitesparse-dev

#Installing Pangolin
git clone https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin
mkdir build
cd build
cmake ../ -DAVFORMAT_INCLUDE_DIR=""
make -j8
cd ../..

if [[ $version == *"15.04"* ]] ; then
    #Up to date OpenNI2 only included in the 14.04 PCL PPA
    git clone https://github.com/occipital/OpenNI2.git
    cd OpenNI2
    make -j8
    cd ..
    
    #15.04 needs ffmpeg to stop OpenCVs build dying, otherwise DBoW won't build (which also won't build with OpenCV3, hurray dependencies!)
    git clone git://source.ffmpeg.org/ffmpeg.git
    cd ffmpeg/
    git reset --hard cee7acfcfc1bc806044ff35ff7ec7b64528f99b1
    ./configure --enable-shared
    make -j8
    sudo make install
    sudo ldconfig
    cd ..
    rm -rf ffmpeg
fi

#Building OpenCV from scratch without Qt and with nonfree
wget http://sourceforge.net/projects/opencvlibrary/files/opencv-unix/2.4.9/opencv-2.4.9.zip
unzip opencv-2.4.9.zip
rm opencv-2.4.9.zip
cd opencv-2.4.9
mkdir build
cd build
cmake -D BUILD_NEW_PYTHON_SUPPORT=OFF -D WITH_OPENCL=OFF -D WITH_OPENMP=ON -D INSTALL_C_EXAMPLES=OFF -D BUILD_DOCS=OFF -D BUILD_EXAMPLES=OFF -D WITH_QT=OFF -D WITH_OPENGL=OFF -D WITH_VTK=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_TESTS=OFF -D WITH_CUDA=OFF -D BUILD_opencv_gpu=OFF ..
make -j8
sudo make install
echo "/usr/local/lib" | sudo tee -a /etc/ld.so.conf.d/opencv.conf
sudo ldconfig
echo "PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig" | sudo tee -a /etc/bash.bashrc
echo "export PKG_CONFIG_PATH" | sudo tee -a /etc/bash.bashrc
source /etc/bash.bashrc
cd ../..
rm -rf opencv-2.4.9

#DLib for place recognition
git clone https://github.com/dorian3d/DLib.git
cd DLib
mkdir build
cd build
cmake ../
make -j8
sudo make install
cd ../..
rm -rf DLib

#DBoW2 for place recognition
git clone https://github.com/dorian3d/DBoW2.git
cd DBoW2
mkdir build
cd build
cmake ../
make -j8
sudo make install
cd ../..
rm -rf DBoW2

#DLoopDetector for place recognition
git clone https://github.com/dorian3d/DLoopDetector.git
cd DLoopDetector
mkdir build
cd build
cmake ../
make -j8
sudo make install
cd ../..
rm -rf DLoopDetector

#iSAM for pose graph optimisation
wget http://people.csail.mit.edu/kaess/isam/isam_v1_7.tgz
tar -xvf isam_v1_7.tgz
rm isam_v1_7.tgz
cd isam_v1_7
cd build
cmake ..
make -j8
sudo make install
cd ../..
rm -rf isam_v1_7

#Actually build Kintinuous
cd ..
mkdir build
cd build
cmake ../src
make -j8
