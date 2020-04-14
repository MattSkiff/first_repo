using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using System.Drawing.Imaging;

namespace KernelConvolution
{
    class Program
    {
        /* 
         * Hi! Thanks for downloading the code associated with the YouTube videos on image filtering.
         * 
         * The code below performs a standard kernel convolution. You can define the kernel yourself
         * near the top of the code. This code is meant as a demonstration, it's not optimised, and
         * is meant only as instruction.
         * 
         * A faster gaussian blur can be found in the other project.
         * I've tested all of this code, but I can't promise there aren't bugs!
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
            if (image.PixelFormat != PixelFormat.Format32bppArgb && image.PixelFormat != PixelFormat.Format32bppRgb)
            {
                return;
            }

            int width = image.Width;
            int height = image.Height;

            // We can't output our values directly to the image, it would affect the convolution. We need an intermediate buffer area.
            uint[] intermediateBuffer = new uint[width * height];

            // System.Bitmap includes SetPixel and GetPixel methods. These are very slow, so it's better to lock
            // the image and access the data directly. Locking prevents C# memory management moving it around.
            BitmapData imageData = image.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.ReadWrite, image.PixelFormat);
           
            uint* ptr = (uint*)imageData.Scan0.ToPointer(); // An unsigned int pointer. This points to the image data in memory, each uint is one pixel ARGB
            int stride = imageData.Stride / 4; // Stride is the width of one pixel row, including any padding. In bytes, /4 converts to 4 byte pixels 

            // Define your kernel here - this one is an approximation of lens focus blur. All 1s would be a mean blur.
            // Note, if your image or kernel is very large, this program will take a long time!
            // For something different, have a look online for an unsharp mask filter.
            
            double[,] kernel = new double[,] { { 0, 0, 0, 1, 0, 0, 0 },
                                               { 0, 1, 1, 1, 1, 1, 0 },
                                               { 0, 1, 1, 1, 1, 1, 0 },
                                               { 1, 1, 1, 1, 1, 1, 1 },
                                               { 0, 1, 1, 1, 1, 1, 0 },
                                               { 0, 1, 1, 1, 1, 1, 0 },
                                               { 0, 0, 0, 1, 0, 0, 0 } };

            // This code only works with square kernels. There's no reason you couldn't adapt it if you like (alter the u / v loops).
            if (kernel.GetLength(0) != kernel.GetLength(1))
            {
                return;
            }
            
            // Kernels should be odd size, else they aren't centred on each pixel
            if (kernel.GetLength(0) % 2 == 0)
            {
                return;
            }

            int kernelRadius = kernel.GetLength(0) / 2; // Integer division, be careful!
            int kernelSize = kernel.GetLength(0);

            byte r, g, b;

            // First pass - convolve
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    // Totals for the convolution
                    double rT = 0.0, gT = 0.0, bT = 0.0, kT = 0.0;

                    for (int v = 0; v < kernelSize; v++)
                    {
                        for (int u = 0; u < kernelSize; u++)
                        {
                            // Current position is cX, cY
                            int cX = x + u - kernelRadius;
                            int cY = y + v - kernelRadius;

                            // We must make sure that the position isn't off the image
                            if (cX < 0 || cX > width - 1 || cY < 0 || cY > height - 1)
                            {
                                continue;
                            }

                            // Obtain current pixel data
                            uint pixel = *(ptr + cY * stride + cX); // Standard formula from the video, px = y * stride + x

                            // Split channels - this uses bit shifting and bitwise AND
                            r = (byte)((0x00FF0000 & pixel) >> 16);
                            g = (byte)((0x0000FF00 & pixel) >> 8);
                            b = (byte)((0x000000FF & pixel));

                            // Add to convolve total
                            rT += r * kernel[u,v];
                            gT += g * kernel[u,v];
                            bT += b * kernel[u,v];
                            kT += kernel[u,v];
                        }
                    }

                    // Compute nearest possible byte values
                    r = (byte)(rT / kT + 0.5);
                    g = (byte)(gT / kT + 0.5);
                    b = (byte)(bT / kT + 0.5);

                    // Combine channels into temporary array -  More bit shifting and bitwise OR.
                    intermediateBuffer[(y * width) + x] = (0xFF000000 | (uint)(r << 16) | (uint)(g << 8) | b);
                }
            }

            // Second pass simply copies buffer back in
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    *(ptr + y * stride + x) = intermediateBuffer[(y * width) + x];
                }
            }

            // Finish with image and save
            image.UnlockBits(imageData);
            image.Save(outputPath);

            TimeSpan duration = DateTime.Now - start;
            Console.WriteLine("Finished in {0} milliseconds.", Math.Round(duration.TotalMilliseconds));
        }
    }
}
