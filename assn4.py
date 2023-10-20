import numpy as np
import cv2
import matplotlib.pyplot as plt

sampleIm = cv2.imread("Sample.jpg", cv2.IMREAD_GRAYSCALE)
capitalIm = cv2.imread("Capitol.jpg", cv2.IMREAD_GRAYSCALE)

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
plt.show()


# QUESTION 3: 

# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

# # ... [Load Image and Compute Centered DFT as before] ...

# def find_largest_magnitudes(F_uv, num_magnitudes):
#     magnitude_spectrum = np.abs(F_uv).copy()
#     center_coords = (magnitude_spectrum.shape[0] // 2, magnitude_spectrum.shape[1] // 2)
#     magnitude_spectrum[center_coords] = 0
#     largest_indices = []
#     for _ in range(num_magnitudes):
#         max_index = np.unravel_index(np.argmax(magnitude_spectrum, axis=None), magnitude_spectrum.shape)
#         largest_indices.append(max_index)
#         magnitude_spectrum[max_index] = 0
#     return largest_indices

# def replace_with_neighbor_average(F_uv, indices):
#     F_uv_modified = F_uv.copy()
#     neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
#     for idx in indices:
#         neighbor_values = [F_uv_modified[idx[0] + n[0], idx[1] + n[1]] for n in neighbors if
#                            0 <= idx[0] + n[0] < F_uv.shape[0] and 0 <= idx[1] + n[1] < F_uv.shape[1]]
#         F_uv_modified[idx] = np.mean(neighbor_values)
#     return F_uv_modified

# def reconstruct_image_by_replacing_magnitudes(F_uv, num_magnitudes):
#     largest_indices = find_largest_magnitudes(F_uv, num_magnitudes)
#     F_uv_modified = replace_with_neighbor_average(F_uv, largest_indices)
#     g_xy_reconstructed = np.abs(np.fft.ifft2(np.fft.ifftshift(F_uv_modified)))
#     return g_xy_reconstructed

# # ... [Compute Centered DFT of Noisy Image as before] ...

# magnitude_numbers = [2, 3, 5, 6]
# reconstructed_images = [reconstruct_image_by_replacing_magnitudes(F_uv_noisy, num) for num in magnitude_numbers]

# fig, ax = plt.subplots(1, len(magnitude_numbers) + 1, figsize=(18, 6))
# ax[0].imshow(img_noisy_gray, cmap='gray')
# ax[0].set_title('Original Noisy Image')
# ax[0].axis('off')

# for i, img in enumerate(reconstructed_images):
#     ax[i+1].imshow(img, cmap='gray')
#     ax[i+1].set_title(f'Reconstructed Image\n({magnitude_numbers[i]} Largest Magnitudes Replaced)')
#     ax[i+1].axis('off')

# plt.subplots_adjust(wspace=0.3)
# plt.show()

