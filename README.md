# Instruction: 

Presented By Group 6 

Amelia Cui & Angelina Duan 

## The following are the instructions to run the program 

it can be invoked at the command line using the following syntax:

// first cd into the project directory 

python fft.py [-m mode] [-i image] [-t test] where the argument is defined as follows:

mode (optional):
- [1] (Default) for fast mode where the image is converted into its FFT form and displayed
- [2] for denoising where the image is denoised by applying an FFT, truncating high
frequencies and then displayed
- [3] for compressing and saving the image
- [4] for plotting the runtime graphs for the report

image (optional): filename of the image we wish to take the DFT of. (Default: the file name of the image given to you for the assignment)
 
test (optional) : the extra test mode added to test FFT algorithm and compare different denoise methods 
- [1] compare our FFT algorithm with the built in algorithm in numpy 
- [2] the comparison amoung three different denoise strategies 
      from left to right 
      "removing high frequencies"
      "removing high frequencies but using different fraction for row and column (but it is not passed as parameter, the fraction can be manipulated by changing the method code ourselves) "
      "threshold everything"

NOTE: if -t 1 / -t 2 is presented in command line, the program runs the test mode regardless the other parameters
Otherwise, it works as the assignment description 

the report link is https://docs.google.com/document/d/1EucHd23aavWrvRxm-l0FIYssupT3rBSTTRU6CJ6A7DA/edit
