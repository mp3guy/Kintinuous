# Kintinuous #

Real-time dense visual SLAM system capable of producing high quality globally consistent point and mesh reconstructions over hundreds of metres in real-time with only a low-cost commodity RGB-D sensor.

# Related Publications #
Please cite the most appropriate of these works (in order of our preference) if you make use of our system in any of your own endeavors:

* **[Real-time Large Scale Dense RGB-D SLAM with Volumetric Fusion](http://thomaswhelan.ie/Whelan14ijrr.pdf)**, *T. Whelan, M. Kaess, H. Johannsson, M.F. Fallon, J. J. Leonard and J.B. McDonald*, IJRR '14
* **[Deformation-based Loop Closure for Large Scale Dense RGB-D SLAM](http://thomaswhelan.ie/Whelan13iros.pdf)**, *T. Whelan, M. Kaess, J.J. Leonard, and J.B. McDonald*, IROS '13
* **[Robust Real-Time Visual Odometry for Dense RGB-D Mapping](http://thomaswhelan.ie/Whelan13icra.pdf)**, *T. Whelan, H. Johannsson, M. Kaess, J.J. Leonard, and J.B. McDonald*, ICRA '13
* **[Kintinuous: Spatially Extended KinectFusion](http://thomaswhelan.ie/Whelan12rssw.pdf)**, *T. Whelan, M. Kaess, M.F. Fallon, H. Johannsson, J. J. Leonard and J.B. McDonald*, RSS RGB-D Workshop '12
* **[A method and system for mapping an environment](https://www.google.com/patents/US9412173)**, *T. Whelan, M. Kaess, J.J. Leonard and J.B. McDonald*, US 9412173 B2

# 1. What do I need to build it? #
* Ubuntu 14.04, 15.04 or 16.04 (Though many other linux distros will work fine)
* CMake
* OpenGL
* [CUDA >= 7.0](https://developer.nvidia.com/cuda-downloads)
* [OpenNI2](https://github.com/occipital/OpenNI2)
* SuiteSparse
* Eigen
* Boost
* zlib
* libjpeg
* [OpenCV](http://sourceforge.net/projects/opencvlibrary/files/opencv-unix/2.4.9/opencv-2.4.9.zip)
* [DLib](https://github.com/dorian3d/DLib) @ 330bdc10576f6bcb55e0bd85cd5296f39ba8811a
* [DBoW2](https://github.com/dorian3d/DBoW2) @ 4a6eed2b3ae35ed6837c8ba226b55b30faaf419d
* [DLoopDetector](https://github.com/dorian3d/DLoopDetector) @ 84bfc56320371bed97cab8aad3aa9561ca931d3f
* [iSAM](http://people.csail.mit.edu/kaess/isam/)
* [PCL](http://pointclouds.org/)
* [Pangolin](https://github.com/stevenlovegrove/Pangolin)

Firstly, add [nVidia's official CUDA repository](https://developer.nvidia.com/cuda-downloads) to your apt sources, then run the following command to pull in most dependencies from the official repos:

```bash
sudo apt-get install -y cmake-qt-gui git build-essential libusb-1.0-0-dev libudev-dev openjdk-7-jdk freeglut3-dev python-vtk libvtk-java libglew-dev cuda-7-5 libsuitesparse-dev openexr
```

This is where things get really bad. Due to libraries constantly changing their APIs and includes, creating build processes that actually last for more than a few months between a couple of Ubuntu versions is extremely difficult. Below are separate instructions for Ubuntu 14.04,  15.04 and 16.04.

**14.04**

Install PCL 1.7 from this PPA:

```bash
sudo add-apt-repository -y ppa:v-launchpad-jochen-sprickerhof-de/pcl
sudo apt-get update
sudo apt-get install -y libpcl-all
```

Then install [OpenCV](http://sourceforge.net/projects/opencvlibrary/files/opencv-unix/2.4.9/opencv-2.4.9.zip), [DLib](https://github.com/dorian3d/DLib), [DBoW2](https://github.com/dorian3d/DBoW2), [DLoopDetector](https://github.com/dorian3d/DLoopDetector), [iSAM](http://people.csail.mit.edu/kaess/isam/) and [Pangolin](https://github.com/stevenlovegrove/Pangolin) from source, in this order.

Why do you have to install OpenCV from source? Because the version in the Ubuntu repos doesn't have the nonfree module, required for SURF descriptors used in the DBoW2. Also, it is strongly recommended you build OpenCV with the following options (in particular, building it with Qt5 might introduce a lot of pain):

```bash
cmake -D BUILD_NEW_PYTHON_SUPPORT=OFF -D WITH_OPENCL=OFF -D WITH_OPENMP=ON -D INSTALL_C_EXAMPLES=OFF -D BUILD_DOCS=OFF -D BUILD_EXAMPLES=OFF -D WITH_QT=OFF -D WITH_OPENGL=OFF -D WITH_VTK=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_TESTS=OFF -D WITH_CUDA=OFF -D BUILD_opencv_gpu=OFF ..
```

If you have trouble building Pangolin, disable ffmpeg support using the following CMake command:

```bash
cmake .. -DAVFORMAT_INCLUDE_DIR=""
```

Once finished you'll have everything you need to build Kintinuous.

**15.04**

On 15.04 PCL is in the official repos:

```bash
sudo apt-get install -y libpcl-dev yasm libvtk5-qt4-dev
```

The version of PCL in the 15.04 repos does not contain OpenNI2. You should build the Occipital maintained version, available [here](https://github.com/occipital/OpenNI2).

As usual, ffmpeg screws things up for everyone in 15.04. You need to build ffmpeg yourself otherwise OpenCV will fail to build. Why do you have to install OpenCV from source? Because the version in the Ubuntu repos doesn't have the nonfree module, required for SURF descriptors used in the DBoW2. Why can't you use OpenCV3? Because DBoW will fail to build. Why can't you disable ffmpeg in OpenCV's build? Because DBoW will fail if the video module isn't built. Build and install ffmpeg as follows:

```bash
git clone git://source.ffmpeg.org/ffmpeg.git
cd ffmpeg/
git reset --hard cee7acfcfc1bc806044ff35ff7ec7b64528f99b1
./configure --enable-shared
make -j8
sudo make install
sudo ldconfig
```

Then build [OpenCV](http://sourceforge.net/projects/opencvlibrary/files/opencv-unix/2.4.9/opencv-2.4.9.zip) with the following options:

```bash
cmake -D BUILD_NEW_PYTHON_SUPPORT=OFF -D WITH_OPENCL=OFF -D WITH_OPENMP=ON -D INSTALL_C_EXAMPLES=OFF -D BUILD_DOCS=OFF -D BUILD_EXAMPLES=OFF -D WITH_QT=OFF -D WITH_OPENGL=OFF -D WITH_VTK=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_TESTS=OFF -D WITH_CUDA=OFF -D BUILD_opencv_gpu=OFF ..
```

Finally, build [DLib](https://github.com/dorian3d/DLib), [DBoW2](https://github.com/dorian3d/DBoW2), [DLoopDetector](https://github.com/dorian3d/DLoopDetector), [iSAM](http://people.csail.mit.edu/kaess/isam/) and [Pangolin](https://github.com/stevenlovegrove/Pangolin) from source, in this order.  If you have trouble building Pangolin, disable ffmpeg support using the following CMake command:

```bash
cmake .. -DAVFORMAT_INCLUDE_DIR=""
```

Afterwards, you will be able to build Kintinuous.

**16.04**

The instructions for 15.04 will work fine except you can't use the apt version of PCL because [someone screwed up vtk](https://bugs.debian.org/cgi-bin/bugreport.cgi?bug=819608). So ensure you purge/remove it completely if you have it installed from apt, then build it manually;

```bash
sudo apt-get install g++ cmake cmake-gui doxygen mpi-default-dev openmpi-bin openmpi-common libflann-dev libeigen3-dev libboost-all-dev libvtk5-qt4-dev libvtk6.2 libvtk5-dev libqhull* libusb-dev libgtest-dev git-core freeglut3-dev pkg-config build-essential libxmu-dev libxi-dev libusb-1.0-0-dev graphviz mono-complete qt-sdk openjdk-7-jdk openjdk-7-jre
git clone https://github.com/PointCloudLibrary/pcl.git
cd pcl
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_GPU=OFF -DBUILD_apps=OFF -DBUILD_examples=OFF ..
make -j8
sudo make install
sudo ldconfig
```

Afterwards, Kintinuous should link without any problems.

# 2. Is there an easier way to build it? #
Understandably, building all of the dependencies seems quite complicated. If you run the *build.sh* script on a fresh clean install of Ubuntu 14.04 or 15.04, enter your password for sudo a few times and wait a few minutes all dependencies will get downloaded and installed and it should build everything correctly. This has not been tested on anything but fresh installs, so I would advise using it with caution if you already have some of the dependencies installed.

# 3. How do I use it? #
There are four build targets:

* *libutil.a* is a small set of utility classes.
* *libfrontend.a* is the main tracking and fusion component.
* *libbackend.a* contains the triangulation, loop closure and deformation components.
* *Kintinuous* is an executable GUI used to run the system.

The GUI (*Kintinuous*) can take a bunch of parameters when launching it from the command line. They are as follows:

* *-c <calibration>* : Loads a camera calibration file specified by either:
 - A depth_intrinsics matrix in OpenCV format (ending .yml, .xml), or
 - A text file containing [fx fy cx cy] or [fx fy cx cy w h]
* *-l <logfile>* : Processes the specified .klg log file.
* *-v <vocab>* : Loads DBoW vocabulary file.
* *-p <poses>* : Loads ground truth poses to use instead of estimated pose.
* *-gpu <gpu>* : Sets which GPU should be used by CUDA.
* *-n <number>* : Number of frames to process.
* *-t <threshold>* : Voxel threshold for volume shifting (default *14*).
* *-cw <weight>* : Removes voxels below this threshold when extracting slices (default *8*).
* *-lt <throttle>* : Disallow loop closures within this time period of the last (default *30*s).
* *-s <size>* : Size of the fusion volume (default *6*m).
* *-dg <sampling>* : Rate of pose sampling for deformation (default *0.8*m).
* *-il <inliers>* : Inlier threshold for RANSAC (default *0.35*).
* *-it <isam>* : Residual threshold for pose graph optimisation (default *10*).
* *-sm* : Static mode (disable volume shifting).
* *-f* : Flip RGB/BGR.
* *-od* : Perform online deformation (required for loop closure).
* *-m* : Enable mesh generation.
* *-no* : Disable overlap of extracted slices.
* *-nos* : Remove overlap when saving map.
* *-r* : Use RGB tracking only.
* *-ri* : Use combined ICP+RGB tracking.
* *-d* : Enable dynamic cube positioning.
* *-dc* : Disable color weighting by angle.
* *-fl* : Subsample pose graph for faster loop closure.
* *-fod* : Enable fast odometry.

Essentially by default *./Kintinuous* will try run off an attached ASUS sensor live. You can provide a .klg log file instead with the -l parameter. You can capture .klg format logs using either [Logger1](https://github.com/mp3guy/Logger1) or [Logger2](https://github.com/mp3guy/Logger2).

# 4. Datasets #

We have provided a sample dataset which you can run easily with Kintinuous for download [here](http://www.cs.nuim.ie/research/vision/data/loop.klg). Launch it as follows:

```bash
./Kintinuous -s 7 -v ../vocab.yml.gz -l loop.klg -ri -fl -od
```

## TUM rgbd dataset ##
[TUM RGBD dataset](http://vision.in.tum.de/data/datasets/rgbd-dataset/download)
1. Download [fr2_desk ](http://vision.in.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_desk.tgz)
2. Uncompress by `tar xvfz rgbd_dataset_freiburg2_desk.tgz`
3. Download `associate.py` from TUM [link](http://vision.in.tum.de/data/datasets/rgbd-dataset/tools)
4. Associate files by `python associate.py ./rgbd_dataset_freiburg2_desk/depth.txt ./rgbd_dataset_freiburg2_desk/rgb.txt > ./rgbd_dataset_freiburg2_desk/associations.txt`
5. Use [png_to_klg](https://github.com/HTLife/png_to_klg) to convert TUM dataset
6. `./pngtoklg -w '~/rgbd_dataset_freiburg2_desk'  -o '~/rgbd_dataset_freiburg2_desk/fr2_desk.klg' -t`

```bash
./Kintinuous -s 7 -v ../vocab.yml.gz -l ~/rgbd_dataset_freiburg2_desk/fr2_desk.klg -ri -fl -od -tum ~/rgbd_dataset_freiburg2_desk/groundtruth.txt -f -s 10 -dg 0.1
```
Evaluate *.poses file in `~/rgbd_dataset_freiburg2_desk` with TUM [evaluate_ate.py](https://svncvpr.in.tum.de/cvpr-ros-pkg/trunk/rgbd_benchmark/rgbd_benchmark_tools/src/rgbd_benchmark_tools/)

# 5. License and Copyright #
The use of the code within this repository and all code within files that make up the software that is Kintinuous is permitted for non-commercial purposes only.  The full terms and conditions that apply to the code within this repository are detailed within the LICENSE.txt file and at [http://www.cs.nuim.ie/research/vision/data/kintinuous/code.php](http://www.cs.nuim.ie/research/vision/data/kintinuous/code.php) unless explicitly stated.  By accessing this repository you agree to comply with these terms.

If you wish to use any of this code for commercial purposes then please email commercialisation@nuim.ie.

Copyright (C) 2015 The National University of Ireland Maynooth and Massachusetts Institute of Technology.

# 6. FAQ #
***What are the hardware requirements?***

A [fast nVidia GPU (1TFLOPS+)](https://en.wikipedia.org/wiki/List_of_Nvidia_graphics_processing_units#GeForce_500_Series), and a fast CPU (something like an i5). If you want to use a non-nVidia GPU you're out of luck.

***The frontend is running fast but the map seems to be lagging behind***

This is because you have a slow CPU. The backend runs completely on the CPU and must process every point extracted from the frontend. This means if your map is very large, or if you're moving very fast, the backend may not be able to keep up. Additionally, turning on meshing when loop closure is enabled is very CPU intensive, but a fast modern processor will cope with this in real-time.

***I saved a map, how can I view it?***

Download [Meshlab](http://meshlab.sourceforge.net/), which can read .ply files. If you only saved the point cloud you'll need to use PCL's viewer for .pcd files.

***The map keeps getting corrupted - tracking is failing - loop closures are incorrect/not working***

Firstly, if you're running live and not processing a log file, ensure you're hitting 30Hz, this is important. Secondly, you cannot move the sensor extremely fast because this violates the assumption behind projective data association. In addition to this, you're probably using a primesense, which means you're suffering from motion blur, unsynchronised cameras and rolling shutter. All of these are aggravated by fast motion and hinder tracking performance.

If you're not getting loop closures and expecting some, or getting false ones, you're at the mercy of DBoW. You can tweak some of the parameters related to it, but over all you're limited by the typical limitations of appearance-based place recognition. Feel free to splice in a different place recognition method. As an aside, [ElasticFusion](https://github.com/mp3guy/ElasticFusion) is much better for very loopy comprehensive scanning, which may suit your application better.

If you notice some weird slicing effect during loop closures, either turn down the volume size or increase the rate at which poses are sampled in the deformation by decreasing the *-dg* parameter.

***Is there a ROS bridge/node?***

No. In fact, if you have ROS installed you're likely to run into some truly horrible build issues.

***This doesn't seem to work like it did in the videos/papers***

A substantial amount of refactoring was carried out in order to open source this system, including rewriting a lot of functionality to avoid certain licenses. Although great care was taken during this process, it is possible that performance regressions were introduced and have not yet been discovered.
