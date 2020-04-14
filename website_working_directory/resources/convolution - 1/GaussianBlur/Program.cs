using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.CompilerServices;

namespace GaussianBlur
{
    class Program
    {
        /* 
         * Hi! Thanks for downloading the code associated with the YouTube videos on image filtering.
         * 
         * The code below performs a reasonably fast 2-pass gaussian blur, just insert some file paths, and choose a standard deviation
         * for your blur. Please note that this code is purely for demonstration purposes. I have made it as clear as I can, with lots
         * of comments, and slightly longer and less efficient code than is possible.
         * 
         * A gaussian blur is a separable filter, so instead of an NxN kernel, we can do Nx1 and 1xN separately. For larger kernels this is
         * much faster.
         * 
         * A more general 2D convolution implementation can be found in the other project. I've tested all of this code, but I can't
         * promise there aren't bugs!
         * 
         * If you are familiar with programming, particularly low level operations like bit shifting, you might like to try and speed
         * the code up. The compiler will speed it up a lot anyway, but I managed 20% or so better performance by making some changes,
         * even after compiler optimisation. Here are some things you could try:
         * 
         * 1) Scale the kernel to use large integers. Integer operations are slightly faster.
         * 2) Pre-compute the left and right (cX), or top and bottom (cY) bounds of the kernel before u and v loops, to avoid the constant bounds checking
         * 3) Test if System.Windows.Media.Imaging.WriteableBitmap is faster than System.Bitmap
         * 4) For really large images or kernels, you could multithread. Try System.Threading.Tasks.Parallel.For for convenience.
        */
        unsafe static void Main(string[] args)
        {
            const string inputPath = @"C:\Directory\image.png";
            const string outputPath = @"C:\Directory\image.out.png";

            DateTime start = DateTime.Now;

            // For this simple example we are using the System.Drawing.Bitmap class.
            Bitmap image = new Bitmap(inputPath);
            
            // This simple code is meant for 32bpp images, it could be adapted to handle other image types.
            // 32ARGB and 32RGB are the same, except in the latter the alpha byte is simply padding.
            if (image.PixelFormat != PixelFormat.Format32bppArgb && image.PixelFormat != PixelFormat.Format32bppRgb)
            {
                return;
            }

            int width = image.Width;
            int height = image.Height;

            // Gaussian blur is best done in two passes, so we need an intermediate image.
            // This doesn't have to be an actual System.Bitmap, any array will do.
            uint[] intermediateBuffer = new uint[width * height];

            // System.Bitmap includes SetPixel and GetPixel methods. These are very slow, so it's better to lock
            // the image and access the data directly. Locking prevents C# memory management moving it around.
            BitmapData imageData = image.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.ReadWrite, image.PixelFormat);
           
            uint* ptr = (uint*)imageData.Scan0.ToPointer(); // An unsigned int pointer. This points to the image data in memory, each uint is one pixel ARGB
            int stride = imageData.Stride / 4; // Stride is the width of one pixel row, including any padding. In bytes, /4 converts to 4 byte pixels 

            double[] kernel = Create1DGaussianKernel(40); // Kernel to be convolved

            int kernelRadius = kernel.Length / 2; // Integer division, be careful!
            int kernelSize = kernel.Length;

            byte r, g, b;

            // First pass - horizontal gaussian blur
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    // Totals for the convolution
                    double rT = 0.0, gT = 0.0, bT = 0.0, kT = 0.0;
                    
                    for (int u = 0; u < kernelSize; u++)
                    {
                        // Current position is cX, y
                        int cX = x + u - kernelRadius;

                        // We must make sure that the position isn't off either end of the image
                        if (cX < 0 || cX > width - 1)
                        {
                            continue;
                        }

                        // Obtain current pixel data
                        uint pixel = *(ptr + y * stride + cX); // Standard formula from the video, px = y * stride + x
                        
                        // Split channels - this uses bit shifting and bitwise AND
                        r = (byte)((0x00FF0000 & pixel) >> 16);
                        g = (byte)((0x0000FF00 & pixel) >> 8);
                        b = (byte)((0x000000FF & pixel));

                        // Add to convolve total
                        rT += r * kernel[u];
                        gT += g * kernel[u];
                        bT += b * kernel[u];
                        kT += kernel[u];
                    }

                    // Compute nearest possible byte values
                    r = (byte)(rT / kT + 0.5);
                    g = (byte)(gT / kT + 0.5);
                    b = (byte)(bT / kT + 0.5);

                    // Combine channels into temporary array -  More bit shifting and bitwise OR.
                    intermediateBuffer[(y * width) + x] = (0xFF000000 | (uint)(r << 16) | (uint)(g << 8) | b);
                }
            }

            // Second pass - vertical gaussian blur
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    // Totals for the convolution
                    double rT = 0.0, gT = 0.0, bT = 0.0, kT = 0.0;

                    for (int v = 0; v < kernelSize; v++)
                    {
                        // Current position is x, cY
                        int cY = y + v - kernelRadius;

                        // We must make sure that the position isn't off either end of the image
                        if (cY < 0 || cY > height - 1)
                        {
                            continue;
                        }

                        // Obtain current pixel data
                        uint pixel = intermediateBuffer[cY * width + x];

                        // Split channels
                        r = (byte)((0x00FF0000 & pixel) >> 16);
                        g = (byte)((0x0000FF00 & pixel) >> 8);
                        b = (byte)((0x000000FF & pixel));

                        // Add to convolve total
                        rT += r * kernel[v];
                        gT += g * kernel[v];
                        bT += b * kernel[v];
                        kT += kernel[v];
                    }

                    // Compute nearest possible byte values
                    r = (byte)(rT / kT + 0.5);
                    g = (byte)(gT / kT + 0.5);
                    b = (byte)(bT / kT + 0.5);

                    // 0xFF000000 here is alpha, not transparent
                    *(ptr + y * stride + x) = (0xFF000000 | (uint)(r << 16) | (uint)(g << 8) | b);
                }
            }

            // Finish with image and save
            image.UnlockBits(imageData);
            image.Save(outputPath);

            TimeSpan duration = DateTime.Now - start;
            Console.WriteLine("Finished in {0} milliseconds.", Math.Round(duration.TotalMilliseconds));
        }

        private static double[] Create1DGaussianKernel(double sd)
        {
            // Normally software will compute the size of the kernel required to adequately capture the standard deviation of the gaussian. 2.5 x sd radius is plenty.
            int radius = (int)Math.Ceiling(sd * 2.5);

            double[] kernel = new double[radius * 2 + 1];
            int kernelPosition = 0;

            // You can see the formula I've used for the Gaussian function at http://en.wikipedia.org/wiki/Normal_distribution.
            double norm = 1 / Math.Sqrt(2 * Math.PI * sd * sd);

            for (int u = -radius; u <= radius; u++)
            {
                kernel[kernelPosition++] = norm * Math.Exp(-(u * u) / (2 * sd * sd));
            }
            
            return kernel;
        }
    }
}
