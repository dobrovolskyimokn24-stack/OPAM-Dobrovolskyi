import numpy as np
import matplotlib.pyplot as plt
import zlib

try:
    from scipy.datasets import camera
except ImportError:
    from scipy.misc import camera

Q50 = np.array([
    [16, 11, 10, 16,  24,  40,  51,  61],
    [12, 12, 14, 19,  26,  58,  60,  55],
    [14, 13, 16, 24,  40,  57,  69,  56],
    [14, 17, 22, 29,  51,  87,  80,  62],
    [18, 22, 37, 56,  68, 109, 103,  77],
    [24, 35, 55, 64,  81, 104, 113,  92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103,  99]
], dtype=np.float64)

def blockify(img):
    h, w = img.shape
    return img.reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)

def unblockify(blocks, h, w):
    return blocks.reshape(h // 8, w // 8, 8, 8).transpose(0, 2, 1, 3).reshape(h, w)

def dct_matrix(N=8):
    D = np.zeros((N, N))
    for k in range(N):
        a = np.sqrt(1/N) if k == 0 else np.sqrt(2/N)
        for n in range(N):
            D[k, n] = a * np.cos(np.pi * (2*n + 1) * k / (2*N))
    return D

D = dct_matrix(8)

def dct2(block):
    return D @ block @ D.T

def idct2(coeffs):
    return D.T @ coeffs @ D

def qtable(q):
    q = max(1, min(99, q))
    S = 5000 / q if q < 50 else 200 - 2 * q
    return np.clip(np.floor((S * Q50 + 50) / 100), 1, 255)

def encode(img, q):
    Q = qtable(q)
    blocks = blockify(img.astype(np.float64) - 128)
    coeffs = np.array([np.round(dct2(b) / Q) for b in blocks])
    return coeffs.astype(np.int16), Q

def decode(coeffs, Q, h, w):
    blocks = np.array([idct2(c * Q) for c in coeffs])
    img = unblockify(blocks, h, w) + 128
    return np.clip(img, 0, 255).round().astype(np.uint8)

def calc_psnr(img1, img2):
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def calc_ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1, img2 = img1.astype(float), img2.astype(float)
    mu1, mu2 = np.mean(img1), np.mean(img2)
    var1, var2 = np.var(img1), np.var(img2)
    cov = np.mean((img1 - mu1) * (img2 - mu2))
    return ((2 * mu1 * mu2 + C1) * (2 * cov + C2)) / ((mu1**2 + mu2**2 + C1) * (var1 + var2 + C2))

def compressed_size(coeffs):
    return len(zlib.compress(coeffs.tobytes()))

if __name__ == "__main__":
    img = camera().astype(np.uint8)
    h, w = img.shape
    qualities = [95, 90, 75, 50, 30, 15, 5]
    psnr_list, ssim_list, size_list = [], [], []

    print(f"{'q':<5} | {'Розмір (байт)':<12} | {'PSNR (дБ)':<10} | {'SSIM':<8}")
    print("-" * 45)

    for q in qualities:
        coeffs, Q = encode(img, q)
        decoded_img = decode(coeffs, Q, h, w)
        size = compressed_size(coeffs)
        psnr = calc_psnr(img, decoded_img)
        ssim = calc_ssim(img, decoded_img)

        size_list.append(size / 1024)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        print(f"{q:<5} | {size:<12} | {psnr:<10.2f} | {ssim:<8.3f}")

        if q in [95, 50, 15, 5]:
            plt.imsave(f"q{q:02d}.png", decoded_img, cmap='gray')

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xlabel('Якість q')
    ax1.set_ylabel('Розмір (КБ)', color='tab:red')
    ax1.plot(qualities, size_list, 'o-', color='tab:red')

    ax2 = ax1.twinx()
    ax2.set_ylabel('PSNR (дБ)', color='tab:blue')
    ax2.plot(qualities, psnr_list, 's--', color='tab:blue')

    plt.title('Залежність якості та розміру від коефіцієнта q')
    plt.savefig('curve.png')
    print("\n[УСПІХ] Готово!")