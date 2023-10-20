import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load the image
image_path = 'Sample.jpg'  # Update path as needed
img = mpimg.imread(image_path)

# Ensure the image is in grayscale if it's not
if len(img.shape) == 3 and img.shape[2] > 1:
    img_gray = np.mean(img, axis=2)
else:
    img_gray = img

# PROBLEM 1 QUESTION 1
# Define the GaussianLowPass function with standard deviations
def GaussianLowPass_adjusted(img, size, sigma1, sigma2):
    D1 = np.zeros([size, size], dtype=float)
    H1 = np.zeros([size, size], dtype=float)
    
    center_u, center_v = size // 2, size // 2  # Center of the frequency rectangle
    
    for u in range(0, size):
        for v in range(0, size):
            # Compute shifted frequencies u and v
            u_shifted = u - center_u
            v_shifted = v - center_v
            
            # Compute Gaussian filter value using sigma1 and sigma2
            H1[u, v] = np.exp(-(u_shifted**2 / (2 * sigma1**2) + v_shifted**2 / (2 * sigma2**2)))
            D1[u, v] = np.sqrt(u_shifted ** 2 + v_shifted ** 2)
    
    return [H1]

# Parameters
sigma1 = 20  # Standard deviation in the u (row) direction
sigma2 = 70  # Standard deviation in the v (column) direction
size = max(img_gray.shape)

# Get Gaussian filter H1 and distance matrix D1 with adjusted function
[H1_adjusted] = GaussianLowPass_adjusted(img_gray, size, sigma1, sigma2)

# Fourier Transform of the image and shift the zero frequency component to the center
F_uv_adjusted = np.fft.fftshift(np.fft.fft2(img_gray))

# Apply the Gaussian filter in the frequency domain (Pixel-wise multiplication)
G_uv_adjusted = F_uv_adjusted * H1_adjusted

# Compute the Inverse Fourier Transform to get the filtered image in spatial domain
g_xy_adjusted = np.abs(np.fft.ifft2(np.fft.ifftshift(G_uv_adjusted)))

# Set up the plots
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# Plot the original image
ax[0].imshow(img_gray, cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('off')

# Plot the Gaussian filter
ax[1].imshow(H1_adjusted, cmap='hot')
ax[1].set_title('Gaussian Low-pass Filter H(u, v)')
ax[1].axis('off')

# Plot the filtered image
ax[2].imshow(g_xy_adjusted, cmap='gray')
ax[2].set_title('Filtered Image')
ax[2].axis('off')

# Adjust spacing between plots
plt.subplots_adjust(wspace=0.3)


# PROBLEM 1 QUESTION 2:

# Define the Butterworth High-Pass filter function
def ButterworthHighPass(img, size, D0, n):
    D = np.zeros([size, size], dtype=float)
    H = np.zeros([size, size], dtype=float)
    
    center_u, center_v = size // 2, size // 2
    
    for u in range(0, size):
        for v in range(0, size):
            # Compute shifted frequencies u and v
            u_shifted = u - center_u
            v_shifted = v - center_v
            
            # Compute distance D and Butterworth filter value using D0 and n
            D[u, v] = np.sqrt(u_shifted ** 2 + v_shifted ** 2)
            H[u, v] = 1 / (1 + (D0 / D[u, v])**(2*n))
    
    return [H]

# Parameters
D0 = 50  # Cutoff frequency
n = 2    # Order of the filter
size = max(img_gray.shape)

# Get Butterworth filter H and distance matrix D
[H] = ButterworthHighPass(img_gray, size, D0, n)

# Fourier Transform of the image and shift the zero frequency component to the center
F_uv = np.fft.fftshift(np.fft.fft2(img_gray))

# Apply the Butterworth filter in the frequency domain
G_uv = F_uv * H

# Compute the Inverse Fourier Transform to get the filtered image in spatial domain
g_xy = np.abs(np.fft.ifft2(np.fft.ifftshift(G_uv)))

# Set up the plots
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# Plot the original image
ax[0].imshow(img_gray, cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('off')

# Plot the Butterworth filter
ax[1].imshow(H, cmap='hot')
ax[1].set_title('Butterworth High-pass Filter H(u, v)')
ax[1].axis('off')

# Plot the filtered image
ax[2].imshow(g_xy, cmap='gray')
ax[2].set_title('Filtered Image')
ax[2].axis('off')

plt.subplots_adjust(wspace=0.3)
plt.show()

# # PROBLEM 2 QUESTION 1:

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
