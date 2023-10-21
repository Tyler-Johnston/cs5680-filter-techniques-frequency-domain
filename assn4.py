import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

sampleIm = cv2.imread("Sample.jpg", cv2.IMREAD_GRAYSCALE)
capitalIm = cv2.imread("Capitol.jpg", cv2.IMREAD_GRAYSCALE)
boyIm = mpimg.imread('boy_noisy.gif')


# PROBLEM 1 QUESTION 1
# The equation provided in the notes only uses a single sigma value, but we were given two
def GaussianLowPass(im, sigma1, sigma2):
    imageHeight, imageWidth = im.shape
    H = np.zeros([imageHeight, imageWidth], dtype=float)

    uCenter, vCenter = imageHeight // 2, imageWidth // 2  # center of the frequency domain
    
    for u in range(imageHeight):
        for v in range(imageWidth):
            # compute shifted frequencies u and v
            uShifted = u - uCenter
            vShifted = v - vCenter
            # compute gaussian filter value using sigma1 and sigma2
            H[u, v] = np.exp(-(uShifted**2 / (2 * sigma1**2) + vShifted**2 / (2 * sigma2**2)))
    return H

sigma1 = 20
sigma2 = 70
gaussianLPFilter = GaussianLowPass(sampleIm, sigma1, sigma2)

# fourier transform of the image / shift the zero frequency component to the center
sampleImFT = np.fft.fftshift(np.fft.fft2(sampleIm))
# apply the Gaussian filter in the frequency domain (Pixel-wise multiplication)
gaussianAppliedLPFT = sampleImFT * gaussianLPFilter
# compute the Inverse Fourier Transform to get the filtered image in spatial domain
gaussianLPInverseFT = np.abs(np.fft.ifft2(np.fft.ifftshift(gaussianAppliedLPFT)))

# plotting
plt.figure(figsize=(10, 5)) # Figure 1
plt.suptitle("Gaussian Low-Pass Filter on Sample.jpg")

plt.subplot(1, 3, 1)
plt.imshow(sampleIm, cmap='gray')
plt.title("Original Image")

plt.subplot(1, 3, 2)
plt.imshow(gaussianLPFilter, cmap='gray')
plt.title("Gaussian Low-Pass Filter")

plt.subplot(1, 3, 3)
plt.imshow(gaussianLPInverseFT, cmap='gray')
plt.title("Filtered Image")
plt.tight_layout()

# PROBLEM 1 QUESTION 2:
def ButterworthHighPass(im, D0, n):
    imageHeight, imageWidth = im.shape
    H = np.zeros([imageHeight, imageWidth], dtype=float)
    
    uCenter, vCenter = imageHeight // 2, imageWidth // 2 # center of the frequency domain
    
    for u in range(imageHeight):
        for v in range(imageWidth):
            uShifted = u - uCenter
            vShifted = v - vCenter
            
            # compute distance and Butterworth filter value using D0 and n
            distance = np.sqrt(uShifted ** 2 + vShifted ** 2)
            # make sure it isn't possible to divide by 0
            if distance != 0:
                H[u, v] = 1 / (1 + (D0 / distance)**(2*n))

    return H

D0 = 50 # Cutoff frequency
n = 2 # Order of the filter
buttersworthHPFilter = ButterworthHighPass(sampleIm, D0, n)

# sampleImFT = np.fft.fftshift(np.fft.fft2(sampleIm)) was already previously defined

# apply the Butterworth filter in the frequency domain
buttersworthAppliedHPFT = sampleImFT * buttersworthHPFilter
# compute the Inverse Fourier Transform to get the filtered image in spatial domain
buttersworthHPInverseFT = np.abs(np.fft.ifft2(np.fft.ifftshift(buttersworthAppliedHPFT)))

# plotting
plt.figure(figsize=(10, 5)) # Figure 2
plt.suptitle("Buttersworth High-Pass Filter on Sample.jpg")

plt.subplot(1, 3, 1)
plt.imshow(sampleIm, cmap='gray')
plt.title("Original Image")

plt.subplot(1, 3, 2)
plt.imshow(buttersworthHPFilter, cmap='gray')
plt.title("Buttersworth High-Pass Filter")

plt.subplot(1, 3, 3)
plt.imshow(buttersworthHPInverseFT, cmap='gray')
plt.title("Filtered Image")
plt.tight_layout()

# PROBLEM 2 QUESTION 1:

# fourier transformation and shift to center for 'sampleIm' and 'capitalIm'
# sampleImFT = np.fft.fftshift(np.fft.fft2(sampleIm)) was already previously defined
capitalImFT = np.fft.fftshift(np.fft.fft2(capitalIm))

# the magnitude of a + bi (where a is real and b is imaginary) is sqrt(a^2 + b^2). this is just the absolute value
sampleImMagnitude = np.abs(sampleImFT)
capitalImMagnitude = np.abs(capitalImFT)

# theta = arctan(b / a) can be represented with np.angle, giving the phase
sampleImPhase = np.angle(sampleImFT)
capitalImPhase = np.angle(capitalImFT)

# log transformations of the magnitude, added by 1 to avoid taking the log of 0
sampleImLog = np.log(sampleImMagnitude + 1)
capitalImLog = np.log(capitalImMagnitude + 1)

# Scaling
scaledSampleImLog = sampleImLog / np.max(sampleImLog)
scaledCapitalImLog = capitalImLog / np.max(capitalImLog)

# plotting
plt.figure(figsize=(12, 5)) # Figure 3
plt.suptitle("Fourier Transform Analysis")

plt.subplot(1, 4, 1)
plt.imshow(scaledSampleImLog, cmap='gray')
plt.title("Magnitude of Sample.jpg")

plt.subplot(1, 4, 2)
plt.imshow(sampleImPhase, cmap='gray', vmin=-np.pi, vmax=np.pi) # phase ranges from -pi to pi
plt.title("Phase of Sample.jpg")

plt.subplot(1, 4, 3)
plt.imshow(scaledCapitalImLog, cmap='gray')
plt.title("Magnitude of Capital.jpg")

plt.subplot(1, 4, 4)
plt.imshow(capitalImPhase, cmap='gray', vmin=-np.pi, vmax=np.pi) # phase ranges from -pi to pi
plt.title("Phase of Capital.jpg")
plt.tight_layout()

# PROBLEM 2 QUESTION 2

# swap magnitude and phase of sample/capital and vice versa
newCapitalFT = sampleImMagnitude * np.exp(1j * capitalImPhase)
newSampleFT = capitalImMagnitude * np.exp(1j * sampleImPhase)
# take inverse fourier transform
newCapitalIm = np.abs(np.fft.ifft2(np.fft.ifftshift(newCapitalFT)))
newSampleIm = np.abs(np.fft.ifft2(np.fft.ifftshift(newSampleFT)))

# plotting
plt.figure(figsize=(10, 5)) # Figure 4
plt.suptitle("Reconstructed capital/sample images after swap")

plt.subplot(1, 2, 1)
plt.imshow(newCapitalIm, cmap='gray')
plt.title("New Capital.jpg")

plt.subplot(1, 2, 2)
plt.imshow(newSampleIm, cmap='gray')
plt.title("New Sample.jpg")
plt.tight_layout()

# QUESTION 3: 

# 1) compute the centered DFT
boyImFT = np.fft.fftshift(np.fft.fft2(boyIm))

# 2) compute the magnitude of the centered DFT image and find the locations (i.e, frequencies) containing the four largest distinct magnitudes
# by excluding the magnitude at the center
boyImMagnitude = np.abs(boyImFT)

def findTopMagnitudes(magnitude, n=4):
    xMid, yMid = magnitude.shape[0] // 2, magnitude.shape[1] // 2
    magnitudeCopy = magnitude.copy()
    # set a 2x2 region around the center to 0, the center values are the highest and not what we want to look at
    magnitudeCopy[xMid-1:xMid+2, yMid-1:yMid+2] = 0
    # .ravel() falttens a 2D array. np.argpartition locates the smallest/largest values without sorting (pretty efficient)
    # the -n param indicates the 'n' largest values are at the end, and [-n:] grabs the last n largest elements only
    indices = np.argpartition(magnitudeCopy.ravel(), -n)[-n:]
    # convert the flattened indices to 2D coordinates
    coordinates = np.column_stack(np.unravel_index(indices, magnitude.shape))
    return coordinates

# get the top 4 magnitude locations and their corresponding pairs, as specified in class
# this suggest the top 8 values need to be computed
topMagnitudes = findTopMagnitudes(boyImMagnitude)

# 3) replace the value at each location with the average of its 8 neighbors
def replaceWithNeighborAverage(magnitude, locations):
    magnitudeCopy = magnitude.copy()
     # For each location, compute the average of its neighbors and replace the value
    for x, y in locations:
        neighbors = [(x-1, y-1), (x-1, y), (x-1, y+1), (x, y-1), (x, y+1), (x+1, y-1), (x+1, y), (x+1, y+1)]
        # ensure neighbors are within the bounds of the image
        validNeighbors = [n for n in neighbors if 0 <= n[0] < magnitude.shape[0] and 0 <= n[1] < magnitude.shape[1]]
        # compute the average of the neighbors
        average = np.mean([magnitude[nx, ny] for nx, ny in validNeighbors])
        # replace the value at the location with the average
        magnitudeCopy[x, y] = average
    return magnitudeCopy

def restoreImage(ft, magnitude, n):

    myTopMagnitudes = findTopMagnitudes(magnitude, n)
    newImMagnitude = replaceWithNeighborAverage(magnitude, myTopMagnitudes)
    # compute the inverse DFT
    phase = np.angle(ft)
    newImFT = newImMagnitude * np.exp(1j * phase)
    restoredImage = np.abs(np.fft.ifft2(np.fft.ifftshift(newImFT)))
    return restoredImage

restoredImageDefault = restoreImage(boyImFT, boyImMagnitude, 4)
restoredImageTwoVals = restoreImage(boyImFT, boyImMagnitude, 2)
restoredImageThreeVals = restoreImage(boyImFT, boyImMagnitude, 3)
restoredImageFiveVals = restoreImage(boyImFT, boyImMagnitude, 5)
restoredImageSixVals = restoreImage(boyImFT, boyImMagnitude, 6)

# plotting original vs default restored noisy boy image
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.imshow(boyIm, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(restoredImageDefault, cmap='gray')
plt.title("Restored Image")
plt.axis('off')
plt.tight_layout()

# plotting the 4 new restored images
plt.figure(figsize=(20, 5))
plt.subplot(1, 4, 1)
plt.imshow(restoredImageTwoVals, cmap='gray')
plt.title("Restored Im: 2 Largest Magnitudes")
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(restoredImageThreeVals, cmap='gray')
plt.title("Restored Im: 3 Largest Magnitudes")
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(restoredImageFiveVals, cmap='gray')
plt.title("Restored Im: 5 Largest Magnitudes")
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(restoredImageSixVals, cmap='gray')
plt.title("Restored Im: 6 Largest Magnitudes")
plt.axis('off')
plt.tight_layout()

print("FIGURE 5 DIFFERENCES:")
print("This difference is pretty obvious - the original contains a LOT of noise. The restored image removes this and the boy is very visible.")
print("The original has those diagonal streaks and the restored removed this.\n")

print("FIGURE 6 DIFFERENCES:")
print("The image with 2 largest distinct magnitudes contains a lot of noise. moving onto 3, we can see that the noise is reduced slightly")
print("Moving onto 5 distinct magnitudes, the noise is significantly reduced and the boy can be clearly seen.\nHowever, moving on to 6 you can see some of the original image is being lost (not by a lot though)")

plt.show()

