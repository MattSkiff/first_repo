using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using System.Drawing.Imaging;

namespace Sobel
{
    class Program
    {
        /* 
         * Hi! Thanks for downloading the code associated with the YouTube videos on image filtering.
         * 
         * The code below performs a sobel edge detection. It then calculates the orientation of the edges, and colours
         * the output using Hue, Saturation and Value. This code is meant as a demonstration, it's not optimised, and
         * is meant only as instruction.
        */
        unsafe static void Main(string[] args)
        {
            const string inputPath = @"C:\Directory\image.png";
            const string outputPath = @"C:\Directory\image.out.png";

            DateTime start = DateTime.Now;

            // For this simple example we are using the System.Drawing.Bitmap class.
            Bitmap image = new Bitmap(inputPath);

            // This simple code is meant for 32bpp images, it could be adapted to handle other image types.
            // 32ARGB and 32RGB are the same, except in the latter the alpha byte is simply padding. We ignore alpha in this code anyway.
            if (image.PixelFormat != PixelFormat.Format32bppArgb 
                && image.PixelFormat != PixelFormat.Format32bppRgb)
            {
                return;
            }

            // Obtain grayscale conversion of the image
            byte[] grayData = ConvertTo8bpp(image);

            int width = image.Width;
            int height = image.Height;

            // Buffers
            byte[] buffer = new byte[9];
            double[] magnitude = new double[width * height]; // Stores the magnitude of the edge response
            double[] orientation = new double[width * height]; // Stores the angle of the edge at that location

            // First pass - convolve sobel operator and calculate orientation. We're using the byte array now, since it's easier.
            for (int y = 1; y < height - 1; y++)
            {
                for (int x = 1; x < width - 1; x++)
                {
                    // Unlike the other Kernel operations, where the radius etc. might change this one is simple so we can hard code
                    // the kernel operations in. Pointer arithmetic would make this slightly faster, but we won't worry about it.
                    int index = y * width + x;

                    // 3x3 window around (x,y)
                    buffer[0] = grayData[index - width - 1];
                    buffer[1] = grayData[index - width];
                    buffer[2] = grayData[index - width + 1];
                    buffer[3] = grayData[index - 1];
                    buffer[4] = grayData[index];
                    buffer[5] = grayData[index + 1];
                    buffer[6] = grayData[index + width - 1];
                    buffer[7] = grayData[index + width];
                    buffer[8] = grayData[index + width + 1];

                    // Sobel horizontal and vertical response
                    double dx = buffer[2] + 2 * buffer[5] + buffer[8] - buffer[0] - 2 * buffer[3] - buffer[6];
                    double dy = buffer[6] + 2 * buffer[7] + buffer[8] - buffer[0] - 2 * buffer[1] - buffer[2];

                    magnitude[index] = Math.Sqrt(dx * dx + dy * dy) / 1141; // 1141 is approximately the max sobel response, we will normalise later anyway

                    // Directional orientation
                    orientation[index] = Math.Atan2(dy, dx) + Math.PI; // Angle is in radians, now from 0 - 2PI. 
                }
            }

            /* Now that we have the magnitude and orientation, we want to combine these into a coloured image for output.
             * The HSV colour model would work well here, hue is the angle of the colour, saturation we keep constant at 1,
             * and value is the magnitude. This should produce an image that is coloured based on edge angle, and whose brightness
             * reflects edge intensity. */

            // System.Bitmap includes SetPixel and GetPixel methods. These are very slow, so it's better to lock
            // the image and access the data directly. Locking prevents C# memory management moving it around.
            BitmapData imageData = image.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.ReadWrite, image.PixelFormat);

            uint* ptr = (uint*)imageData.Scan0.ToPointer(); // An unsigned int pointer. This points to the image data in memory, each uint is one pixel ARGB
            int stride = imageData.Stride / 4; // Stride is the width of one pixel row, including any padding. In bytes, /4 converts to 4 byte pixels 

            byte r, g, b;

            // We want to scale magnitude from 0 - 1, because it's unlikely any magnitude would reach the theoretical maximum value
            // C#, like other high level languages, contains various list extension methods for ease of use.
            double magnitudeMax = magnitude.Max();
            
            // Combine 
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int index = y * width + x;

                    // C# functions can make use of an out parameter. This works much like C/C++ address pointer (&), except it is compiler enforced
                    // Here we can use it to essentially return three values
                    double hue = orientation[index];
                    double val = magnitude[index] / magnitudeMax; // This will still highlight mostly very bright edges, to highlight more try Math.Sqrt(magnitude[index] / magnitudeMax)
                    
                    HSVtoRGB(hue, 1, val, out r, out g, out b); // Using a saturation of 0 will make the image grayscale i.e. regular sobel.

                    // Combine rgb back into the output image
                    *(ptr + y * stride + x) = (0xFF000000 | (uint)(r << 16) | (uint)(g << 8) | b);
                }
            }

            // Finish with image and save
            image.UnlockBits(imageData);
            image.Save(outputPath);

            TimeSpan duration = DateTime.Now - start;
            Console.WriteLine("Finished in {0} milliseconds.", Math.Round(duration.TotalMilliseconds));
        }

        unsafe public static byte[] ConvertTo8bpp(Bitmap image)
        {
            int width = image.Width;
            int height = image.Height;

            byte[] grayData = new byte[width * height];

            BitmapData imageData = image.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.ReadOnly, image.PixelFormat);
           
            uint* ptr = (uint*)imageData.Scan0.ToPointer();
            int inputStride = imageData.Stride / 4;

            byte r, g, b;

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    // Obtain current pixel data
                    uint pixel = *(ptr + y * inputStride + x); // Standard formula from the video, px = y * stride + x

                    // Split channels - this uses bit shifting and bitwise AND
                    r = (byte)((0x00FF0000 & pixel) >> 16);
                    g = (byte)((0x0000FF00 & pixel) >> 8);
                    b = (byte)((0x000000FF & pixel));

                    byte gray = (byte)(0.2126 * r + 0.7152 * g + 0.0722 * b);
                    grayData[y * width + x] = gray;
                }
            }

            image.UnlockBits(imageData);
           
            return grayData;
        }

        private static void HSVtoRGB(double h, double s, double v, out byte r, out byte g, out byte b)
        {
            // h_ 0-6
            double h_ = h / (2 * Math.PI) * 6;

            double c = s * v;
            double x = c * (1 - Math.Abs((h_ % 2) - 1));
            double r_, g_, b_;
            if (h_ < 1)
            {
                r_ = c;
                g_ = x;
                b_ = 0;
            }
            else if (h_ < 2)
            {
                r_ = x;
                g_ = c;
                b_ = 0;
            }
            else if (h_ < 3)
            {
                r_ = 0;
                g_ = c;
                b_ = x;
            }
            else if (h_ < 4)
            {
                r_ = 0;
                g_ = x;
                b_ = c;
            }
            else if (h_ < 5)
            {
                r_ = x;
                g_ = 0;
                b_ = c;
            }
            else
            {
                r_ = c;
                g_ = 0;
                b_ = x;
            }

            double m = v - c;

            r_ += m;
            g_ += m;
            b_ += m;

            r = (byte)(r_ * 255);
            g = (byte)(g_ * 255);
            b = (byte)(b_ * 255);
        }
    }
}
