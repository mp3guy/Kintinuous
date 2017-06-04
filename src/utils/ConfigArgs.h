/*
 * This file is part of Kintinuous.
 *
 * Copyright (C) 2015 The National University of Ireland Maynooth and 
 * Massachusetts Institute of Technology
 *
 * The use of the code within this file and all code within files that 
 * make up the software that is Kintinuous is permitted for 
 * non-commercial purposes only.  The full terms and conditions that 
 * apply to the code within this file are detailed within the LICENSE.txt 
 * file and at <http://www.cs.nuim.ie/research/vision/data/kintinuous/code.php> 
 * unless explicitly stated.  By downloading this file you agree to 
 * comply with these terms.
 *
 * If you wish to use any of this code for commercial purposes then 
 * please email commercialisation@nuim.ie.
 */

#ifndef CONFIGARGS_H_
#define CONFIGARGS_H_

#include <unistd.h>
#include <cassert>
#include <string>
#include <pcl/console/parse.h>
#include <boost/filesystem.hpp>

class ConfigArgs
{
    public:
        static const ConfigArgs & get(int argc = 0, char ** argv = 0)
        {
            static const ConfigArgs instance(argc, argv);
            return instance;
        }

        static void usage(const std::string argv0)
        {
            const std::string message = "Usage: " + argv0 + " [Options]\n"
                                        "Kintinuous dense mapping system\n"
                                        "\n"
                                        "Options:\n"
                                        "    -c <calibration> : Loads a camera calibration file specified by either:\n"
                                        "        a) a depth_intrinsics matrix in OpenCV format (ending .yml, .xml), or\n"
                                        "        b) a text file containing [fx fy cx cy] or [fx fy cx cy w h]\n"
                                        "    -l <logfile> : Processes the specified .klg log file\n"
                                        "    -v <vocab> : Loads DBoW vocabulary file\n"
                                        "    -p <poses> : Loads ground truth poses to use instead of estimated pose\n"
                                        "    -gpu <gpu> : Sets which GPU should be used by CUDA\n"
                                        "    -n <number> : Number of frames to process\n"
                                        "    -t <threshold> : Voxel threshold for volume shifting (default 14)\n"
                                        "    -cw <weight> : Removes voxels below this threshold when extracting slices (default 8)\n"
                                        "    -lt <throttle> : Disallow loop closures within this time period of the last (default 30s)\n"
                                        "    -s <size> : Size of the fusion volume (default 6m)\n"
                                        "    -dg <sampling> : Rate of pose sampling for deformation (default 0.8m)\n"
                                        "    -il <inliers> : Inlier threshold for RANSAC (default 0.35)\n"
                                        "    -it <isam> : Residual threshold for pose graph optimisation (default 10)\n"
                                        "    -sm : Static mode (disable volume shifting)\n"
                                        "    -f : Flip RGB/BGR\n"
                                        "    -od : Perform online deformation (required for loop closure)\n"
                                        "    -m : Enable mesh generation\n"
                                        "    -no : Disable overlap of extracted slices\n"
                                        "    -nos : Remove overlap when saving map\n"
                                        "    -r : Use RGB tracking only\n"
                                        "    -ri : Use combined ICP+RGB tracking\n"
                                        "    -d : Enable dynamic cube positioning\n"
                                        "    -dc : Disable color weighting by angle\n"
                                        "    -fl : Subsample pose graph for faster loop closure\n"
                                        "    -fod : Enable fast odometry\n"
                                        "    -h, --help : Display this help message and exit\n"
                                        "    -tum : path of groundtruth.txt provided by TUM RGBD dataset to load initial camera pose\n"
                                        "\n"
                                        "Example: " + argv0 + " -s 7 -v ../vocab.yml.gz -l loop.klg -ri -fl -od";

            std::cout << message << std::endl;
        }

        //Parameters
        std::string calibrationFile;
        std::string logFile;
        std::string vocabFile;
        std::string trajectoryFile;

        std::string tumGT;
        long long unsigned int utime;
        float x, y, z, qx, qy, qz, qw;

        int gpu;
        int voxelShift;
        int totalNumFrames;
        int weightCull;
        int loopThrottle;

        float volumeSize;
        float denseSamplingRate;
        float inlierRatio;
        float isamThresh;

        bool staticMode;
        bool flipColors;
        bool enableMeshGenerator;
        bool extractOverlap;
        bool saveOverlap;
        bool useRGBD;
        bool useRGBDICP;
        bool dynamicCube;
        bool onlineDeformation;
        bool disableColorAngleWeight;
        bool incrementalMesh;
        bool fastLoops;
        bool fastOdometry;

        //Auto generated
        std::string saveFile;

    private:
        ConfigArgs(int argc, char ** argv)
         : gpu(0),
           voxelShift(14),
           totalNumFrames(0),
           weightCull(8),
           loopThrottle(30),
           volumeSize(6.0f),
           denseSamplingRate(0.8f),
           inlierRatio(0.35f),
           isamThresh(10.0f)
        {
            assert(argc && argv);

            bool help = pcl::console::find_switch(argc, argv, "-h") || pcl::console::find_switch(argc, argv, "--help");

            if(help)
            {
                ConfigArgs::usage(argv[0]);
                exit(0);
            }

            pcl::console::parse_argument(argc, argv, "-c", calibrationFile);
            pcl::console::parse_argument(argc, argv, "-l", logFile);
            pcl::console::parse_argument(argc, argv, "-v", vocabFile);
            pcl::console::parse_argument(argc, argv, "-p", trajectoryFile);
            pcl::console::parse_argument(argc, argv, "-tum", tumGT);
            if(tumGT.size())
            {
                loadTUM_GT_initialpose();
            }

            pcl::console::parse_argument(argc, argv, "-gpu", gpu);
            pcl::console::parse_argument(argc, argv, "-t", voxelShift);
            pcl::console::parse_argument(argc, argv, "-n", totalNumFrames);
            pcl::console::parse_argument(argc, argv, "-cw", weightCull);
            pcl::console::parse_argument(argc, argv, "-lt", loopThrottle);

            pcl::console::parse_argument(argc, argv, "-s", volumeSize);
            pcl::console::parse_argument(argc, argv, "-dg", denseSamplingRate);
            pcl::console::parse_argument(argc, argv, "-il", inlierRatio);
            pcl::console::parse_argument(argc, argv, "-it", isamThresh);

            staticMode = pcl::console::find_switch(argc, argv, "-sm");
            flipColors = pcl::console::find_switch(argc, argv, "-f");
            onlineDeformation = pcl::console::find_switch(argc, argv, "-od") && vocabFile.length();
            enableMeshGenerator = pcl::console::find_switch(argc, argv, "-m");
            extractOverlap = !pcl::console::find_switch(argc, argv, "-no");
            saveOverlap = !pcl::console::find_switch(argc, argv, "-nos");
            useRGBD = pcl::console::find_switch(argc, argv, "-r");
            useRGBDICP = pcl::console::find_switch(argc, argv, "-ri");
            dynamicCube = pcl::console::find_switch(argc, argv, "-d");
            disableColorAngleWeight = pcl::console::find_switch(argc, argv, "-dc");
            fastLoops = pcl::console::find_switch(argc, argv, "-fl") && onlineDeformation;
            incrementalMesh = enableMeshGenerator && onlineDeformation;
            fastOdometry = pcl::console::find_switch(argc, argv, "-fod");

            if(voxelShift < 1 || voxelShift > 16)
            {
                std::cout << "Voxel shift must between 1 and 16, correcting to ";

                voxelShift = std::max(1, std::min(voxelShift, 16));

                std::cout << voxelShift << std::endl;
            }

            if(!logFile.size())
            {
                char buf[256];
                int length = readlink("/proc/self/exe", buf, sizeof(buf));

                std::string currentVal;
                currentVal.append((char *)&buf, length);

                saveFile = currentVal;
            }
            else
            {
                saveFile = logFile;
            }
        }

        void loadTUM_GT_initialpose()
        {
            assert(boost::filesystem::exists(tumGT.c_str()));
            FILE *fp = fopen(tumGT.c_str(), "r");


            char buffer[255];

            int iSkip = 3;
            for(int i=0; i<iSkip; i++)
            {
                fgets(buffer, 255, (FILE*) fp);
            }

            //Read first camera pose
            if(fgets(buffer, 255, (FILE*) fp)) {
                printf("%s\n", buffer);

                int n = sscanf(buffer, "%llu %f %f %f %f %f %f %f", &utime, &x, &y, &z, &qx, &qy, &qz, &qw);

                assert(n == 8);
            }

            fclose(fp);
        }
};

#endif /* CONFIGARGS_H_ */
