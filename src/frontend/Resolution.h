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


#ifndef RESOLUTION_H_
#define RESOLUTION_H_

class Resolution
{
    public:
        static const Resolution & get(int width = 0, int height = 0)
        {
            static const Resolution instance(width, height);
            return instance;
        }

        const int & width() const
        {
            return imgWidth;
        }

        const int & height() const
        {
            return imgHeight;
        }

        const int & cols() const
        {
            return imgWidth;
        }

        const int & rows() const
        {
            return imgHeight;
        }

        const int & numPixels() const
        {
            return imgNumPixels;
        }

    private:
        Resolution(int width, int height)
         : imgWidth(width),
           imgHeight(height),
           imgNumPixels(width * height)
        {
            assert(width > 0 && height > 0);
        }

        const int imgWidth;
        const int imgHeight;
        const int imgNumPixels;
};

#endif /* RESOLUTION_H_ */
