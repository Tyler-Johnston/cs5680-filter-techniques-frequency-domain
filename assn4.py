import numpy as np
import cv2
import matplotlib.pyplot as plt
import pywt
import matplotlib.image as mpimg
from skimage.util import random_noise # for problem 5, this is a built-in function that adds gaussian white noise

sampleIm = cv2.imread("Sample.jpg", cv2.IMREAD_GRAYSCALE)
capitalIm = cv2.imread("Capitol.jpg", cv2.IMREAD_GRAYSCALE)
lenaIm = cv2.imread("Lena.jpg", cv2.IMREAD_GRAYSCALE)
boyIm = mpimg.imread('boy_noisy.gif') # use mpimg.imread for .gif images, as cv2 as issues reading this

def PSNR(original, processed):
    mse = np.mean((original - processed) ** 2)
    # ensures there isn't a divsion of zero   
    if mse == 0:
        return float('inf')
    maximumPossibleIntensity = 255
    psnr = 10 * np.log10((maximumPossibleIntensity ** 2) / mse)
    return psnr

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

# PROBLEM 3: 

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
plt.figure(figsize=(15, 5))
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
print("Moving onto 5 distinct magnitudes, the noise is significantly reduced and the boy can be clearly seen.\nHowever, moving on to 6 you can see some of the original image is being lost (not by a lot though)\n")

# PROBLEM 4

# call the built-in function to compare lenaIm and the wavelet restoredLena
maxLevel = pywt.dwt_max_level(lenaIm.shape[0], pywt.Wavelet('db2'))
# apply a maximum-level "db2" wavelet decomposition
coeffs = pywt.wavedec2(lenaIm, 'db2', level=maxLevel)
# apply the inverse wavelet transform to restore the image
restoredLena = pywt.waverec2(coeffs, 'db2')

# comparing the matrix of lenaIm and restoredLena, there was exactly 1 pixel difference
# round the values and clip them to the valid range ensures they are exactly the same
# although in real-applications, this isn't necessary as this difference is incredibly negligible
restoredLenaRounded = np.round(restoredLena).astype(np.uint8)
restoredLenaClipped = np.clip(restoredLenaRounded, 0, 255)

if np.array_equal(lenaIm, restoredLenaClipped):
    print("The original and the restored images (of lena / restored lena) are the same.\n")
else:
    print("The original and the restored images (of lena / restored lena) are different.\n")

# 3-level decomposition
coeffs = pywt.wavedec2(lenaIm, 'db2', level=3)

CA3 = coeffs[0]
CH3, CV3, CD3 = coeffs[1]
CH2, CV2, CD2 = coeffs[2]
CH1, CV1, CD1 = coeffs[3]

# SET 1:
CA3Copy = np.copy(CA3)
CA3Height, CA3Width = CA3.shape

# set the values of each 4x4 non-overlapping block in CA3 to its average
for i in range(0, CA3Height, 4):
    for j in range(0, CA3Width, 4):
        blockAvg = np.mean(CA3[i:i+4, j:j+4])
        CA3Copy[i:i+4, j:j+4] = blockAvg

set1Coeffs = coeffs.copy()
set1Coeffs[0] = CA3Copy
set1RestoredImage = pywt.waverec2(set1Coeffs, 'db2')

# SET 2:
set2Coeffs = coeffs.copy()
CH1Copy = np.copy(CH1)
CH1Height, CH1Width = CH1.shape

for i in range(CH1Height):
    for j in range(CH1Width):
        CH1Copy[i][j] = 0

set2Coeffs[3] = CH1Copy, CV1, CD1
set2RestoredImage = pywt.waverec2(set2Coeffs, 'db2')

# SET 3:
set3Coeffs = coeffs.copy()
CD2Copy = np.copy(CD2)
CD2Height, CD2Width = CD2.shape

for i in range(CD2Height):
    for j in range(CD2Width):
        CD2Copy[i][j] = 0

set3Coeffs[2] = CH2, CV2, CD2Copy
set3RestoredImage = pywt.waverec2(set3Coeffs, 'db2')

# SET 4:
set4Coeffs = coeffs.copy()
CV3Copy = np.copy(CV3)
CV3Height, CV3Width = CV3.shape

for i in range(CV3Height):
    for j in range(CV3Width):
        CV3Copy[i][j] = 0

set4Coeffs[1] = CH3, CV3Copy, CD3
set4RestoredImage = pywt.waverec2(set4Coeffs, 'db2')

# differences
print("ORIGINAL VS SET 1 (CA3):")
print("The difference is visually apparent. The original lady is clearly visible and sharply defined. However, set 1 displays distinct blurry blocks causing a loss of clarity")
print("This occured because CA3 only contains the low-level frequencies of the origninal image (lower level of detail), and taking the average of those 4x4 non-overlapping blocks and setting those values caused a blurry blocky image\n")

print("ORIGINAL VS SET 2 (CH1):")
print("Set 2 exhibits a subtle reduction in horizontal details compared to the original.\nAlthough there isn't a ton of obvious changes, modifying CH1 should be softening the horizontal edges. At level-1, this didn't cause an obvious change")
print("The horizontal details at level 1 are not the primary features of the image. If this were a different image with pronunced horizontal features, its likely that the changes would have been more obvious\n")

print("ORIGINAL VS SET 3 (CD2):")
print("Similar deal as set 2. There's a slight reduction in diagonal details/edges.\nWhile its not incredibly obvious, this happens from the zeroing of the CD2 coefficients.")
print("This effect suggests that the diagonal details at level 2 are not a dominant feature of 'lena'. If this were a different image with pronunced diagonal features, its likely that the changes would have been more obvious\n")

print("ORIGINAL VS SET 4 (CV3):")
print("The changes in set 4 are relatively more noticeable. Certain regions, particularly around edges, appear rougher.\nThis is a result of setting the CV3 coefficients, which capture vertical details at level 3, to zero. This indicates that the image had significant vertical details at this level.\n")

# plotting
plt.figure(figsize=(5, 5)) # Figure 7
plt.suptitle("4x4 average approximation")
plt.subplot(1, 1, 1)
plt.imshow(set1RestoredImage, cmap='gray')
plt.title("Set #1:")
plt.axis("off")

plt.figure(figsize=(5, 5)) # Figure 8
plt.suptitle("1st level horizontal details to 0")
plt.subplot(1, 1, 1)
plt.imshow(set2RestoredImage, cmap='gray')
plt.title("Set #2:")
plt.axis("off")

plt.figure(figsize=(5, 5)) # Figure 9
plt.suptitle("2nd level diagonal details to 0")
plt.subplot(1, 1, 1)
plt.imshow(set3RestoredImage, cmap='gray')
plt.title("Set #3:")
plt.axis("off")

plt.figure(figsize=(5, 5)) # Figure 10
plt.suptitle("3rd level vertical details to 0")
plt.subplot(1, 1, 1)
plt.imshow(set4RestoredImage, cmap='gray')
plt.title("Set #4:")
plt.axis("off")

# PROBLEM 5

noisyLenaOriginal = random_noise(lenaIm, mode='gaussian', mean=0, var=0.01) * 255
noisyLenaOriginal = np.clip(noisyLenaOriginal, 0, 255).astype(np.uint8)
cv2.imwrite("NoisyLena.bmp", noisyLenaOriginal)

# Denoising Method 1
noisyLena = cv2.imread("NoisyLena.bmp", cv2.IMREAD_GRAYSCALE)
myCoeffs = pywt.wavedec2(noisyLena, 'db2', level=3)

CA3 = myCoeffs[0]
HL3, LH3, HH3 = myCoeffs[1]
HL2, LH2, HH2 = myCoeffs[2]
HL1, LH1, HH1 = myCoeffs[3]

def denoisingMethod(HHX, LHX, HLX, method1=True):

    if method1:
        noiseStdDev = np.median(np.abs(HH1)) / 0.6745
    else: 
        combinedCoeffs = np.hstack((LH1.ravel(), HL1.ravel(), HH1.ravel()))
        noiseStdDev = np.median(np.abs(combinedCoeffs)) / 0.6745

    M = np.prod(LHX.shape) + np.prod(HLX.shape) + np.prod(HHX.shape)
    t = noiseStdDev * np.sqrt(2 * np.log(M))

    LHXCopy = np.copy(LHX)
    LHXCopy[LHX >= t] -= t
    LHXCopy[LHX <= -t] += t
    LHXCopy[np.abs(LHX) < t] = 0

    HLXCopy = np.copy(HLX)
    HLXCopy[HLX >= t] -= t
    HLXCopy[HLX <= -t] += t
    HLXCopy[np.abs(HLX) < t] = 0

    HHXCopy = np.copy(HHX)
    HHXCopy[HHX >= t] -= t
    HHXCopy[HHX <= -t] += t
    HHXCopy[np.abs(HHX) < t] = 0

    return (HLXCopy, LHXCopy, HHXCopy)

# METHOD 1
HL1Copy1, LH1Copy1, HH1Copy1 = denoisingMethod(HH1, LH1, HL1)
HL2Copy1, LH2Copy1, HH2Copy1 = denoisingMethod(HH2, LH2, HL2)
HL3Copy1, LH3Copy1, HH3Copy1 = denoisingMethod(HH3, LH3, HL3)

denoisedCoeffs1 = [CA3, (HL3Copy1, LH3Copy1, HH3Copy1), (HL2Copy1, LH2Copy1, HH2Copy1), (HL1Copy1, LH1Copy1, HH1Copy1)]
denoisedLena1 = pywt.waverec2(denoisedCoeffs1, 'db2')

# METHOD 2
HL1Copy2, LH1Copy2, HH1Copy2 = denoisingMethod(HH1, LH1, HL1, False)
HL2Copy2, LH2Copy2, HH2Copy2 = denoisingMethod(HH2, LH2, HL2, False)
HL3Copy2, LH3Copy2, HH3Copy2 = denoisingMethod(HH3, LH3, HL3, False)

denoisedCoeffs2 = [CA3, (HL3Copy2, LH3Copy2, HH3Copy2), (HL2Copy2, LH2Copy2, HH2Copy2), (HL1Copy2, LH1Copy2, HH1Copy2)]
denoisedLena2 = pywt.waverec2(denoisedCoeffs2, 'db2')

method1PSNR = PSNR(lenaIm, denoisedLena1)
method2PSNR = PSNR(lenaIm, denoisedLena2)

print("Method 1 PSNR: ", method1PSNR)
print("Method 2 PSNR: ", method2PSNR)

print("The two denoised images are nearly identical. However, method 1 has a ever-so-slightly higher PSNR, which indicates it is ever-so-slightly closer to the original")

plt.figure(figsize=(10, 5)) # Figure 11
plt.suptitle("Denoisng Methods")
plt.subplot(1, 2, 1)
plt.imshow(denoisedLena1, cmap='gray')
plt.title("Method 1")
plt.axis("off")

plt.subplot(1, 2, 2) # Figure 12
plt.imshow(denoisedLena2, cmap='gray')
plt.title("Method 2")
plt.axis("off")
plt.show()