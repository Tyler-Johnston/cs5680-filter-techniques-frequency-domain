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
gaussianLPFT = np.fft.fftshift(np.fft.fft2(sampleIm))
# apply the Gaussian filter in the frequency domain (Pixel-wise multiplication)
gaussianAppliedLPFT = gaussianLPFT * gaussianLPFilter
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

# fourier Transform of the image and shift the zero frequency component to the center
buttersworthHPFT = np.fft.fftshift(np.fft.fft2(sampleIm))
# apply the Butterworth filter in the frequency domain
buttersworthAppliedHPFT = buttersworthHPFT * buttersworthHPFilter
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


# # Load the images
# img2 = mpimg.imread('Capitol.jpg')

# img_gray1 = np.mean(img, axis=2) if len(img.shape) == 3 else img
# # Ensure the images are in grayscale
# img_gray2 = np.mean(img2, axis=2) if len(img2.shape) == 3 else img2

# # Fourier Transform and shift zero frequency component to the center
# F_uv1 = np.fft.fftshift(np.fft.fft2(img_gray1))
# F_uv2 = np.fft.fftshift(np.fft.fft2(img_gray2))

# # Extract magnitude and phase
# magnitude1 = np.abs(F_uv1)
# phase1 = np.angle(F_uv1)
# magnitude2 = np.abs(F_uv2)
# phase2 = np.angle(F_uv2)

# # Log transformation of magnitude
# log_magnitude1 = np.log(magnitude1 + 1)
# log_magnitude2 = np.log(magnitude2 + 1)

# # Scaling
# scaled_log_magnitude1 = log_magnitude1 / np.max(log_magnitude1)
# scaled_log_magnitude2 = log_magnitude2 / np.max(log_magnitude2)

# # ... [VISUALIZE MAGNITUDE AND PHASE SPECTRUMS HERE] ...

# # Set up the plots

# fig, ax = plt.subplots(2, 2, figsize=(12, 12))



# # Plot the magnitude and phase of the Fourier-transformed "Sample" image

# ax[0, 0].imshow(scaled_log_magnitude1, cmap='gray')

# ax[0, 0].set_title('Magnitude Spectrum of Sample')

# ax[0, 0].axis('off')

# ax[0, 1].imshow(phase1, cmap='gray')

# ax[0, 1].set_title('Phase Spectrum of Sample')

# ax[0, 1].axis('off')



# # Plot the magnitude and phase of the Fourier-transformed "Capitol" image

# ax[1, 0].imshow(scaled_log_magnitude2, cmap='gray')
# ax[1, 0].set_title('Magnitude Spectrum of Capitol')
# ax[1, 0].axis('off')
# ax[1, 1].imshow(phase2, cmap='gray')
# ax[1, 1].set_title('Phase Spectrum of Capitol')
# ax[1, 1].axis('off')


# # PROBLEM 2 QUESTION 2

# # Swap phase and perform inverse Fourier transform
# G_uv1 = magnitude1 * np.exp(1j * phase2)
# G_uv2 = magnitude2 * np.exp(1j * phase1)
# g_xy1 = np.abs(np.fft.ifft2(np.fft.ifftshift(G_uv1)))
# g_xy2 = np.abs(np.fft.ifft2(np.fft.ifftshift(G_uv2)))

# fig, ax = plt.subplots(1, 2, figsize=(12, 6))


# # Plot the reconstructed image using the magnitude of Sample and phase of Capitol

# ax[0].imshow(g_xy1, cmap='gray')
# ax[0].set_title('Reconstructed Image\n(Magnitude: Sample, Phase: Capitol)')
# ax[0].axis('off')

# # Plot the reconstructed image using the magnitude of Capitol and phase of Sample
# ax[1].imshow(g_xy2, cmap='gray')
# ax[1].set_title('Reconstructed Image\n(Magnitude: Capitol, Phase: Sample)')
# ax[1].axis('off')

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

