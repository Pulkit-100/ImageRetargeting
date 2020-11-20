import warnings
import numpy as np
import cv2
from numba import jit
from scipy import ndimage as ndi

warnings.filterwarnings("ignore")


SEAM_COLOR = np.array([255, 200, 200])    # seam visualization color (BGR)
SHOULD_DOWNSIZE = True                    # if True, downsize image for faster carving
DOWNSIZE_WIDTH = 600                      # resized image width if SHOULD_DOWNSIZE is True
DOWNSIZE_HEIGHT = 600
ENERGY_MASK_CONST = 100000.0              # large energy value for protective masking
MASK_THRESHOLD = 10                       # minimum pixel intensity for binary mask


def visualize(im, boolmask=None, rotate=False):
    vis = im.astype(np.uint8)
    # if boolmask is not None:
    #     vis[np.where(boolmask == False)] = SEAM_COLOR
    #     pass
    if rotate:
        vis = rotate_image(vis, False)
    cv2.imshow("processing", vis)
    cv2.waitKey(1)
    return vis


def resize(image, width = None, height = None):
    if width:
        dim = None
        h, w = image.shape[:2]
        dim = (width, int(h * width / float(w)))
        return cv2.resize(image, dim)
    if height:
        h, w = image.shape[:2]
        dim = (int(w*height/float(h)), height)
        return cv2.resize(image, dim)


def rotate_image(image, clockwise):
    k = 1 if clockwise else 3
    return np.rot90(image, k)


def backward_energy(im):
    """
    Simple gradient magnitude energy map.
    """

    xgrad = ndi.convolve1d(im, np.array([1, 0, -1]), axis=1, mode='wrap')
    ygrad = ndi.convolve1d(im, np.array([1, 0, -1]), axis=0, mode='wrap')

    grad_mag = np.sqrt(np.sum(xgrad ** 2, axis=2) + np.sum(ygrad ** 2, axis=2))

    return grad_mag


@jit
def remove_seam(im, boolmask):
    h, w = im.shape[:2]
    boolmask3c = np.stack([boolmask] * 3, axis=2)
    return im[boolmask3c].reshape((h, w - 1, 3))


@jit
def add_seam(im, seam_idx):

    h, w = im.shape[:2]
    print(h,w)
    output = np.zeros((h, w + 1, 3))
    for row in range(h):
        col = seam_idx[row]
        for ch in range(3):
            if col == 0:
                p = np.average(im[row, col: col + 2, ch])
                output[row, col, ch] = im[row, col, ch]
                output[row, col + 1, ch] = p
                output[row, col + 1:, ch] = im[row, col:, ch]
            else:
                p = np.average(im[row, col - 1: col + 1, ch])
                output[row, : col, ch] = im[row, : col, ch]
                output[row, col, ch] = p
                output[row, col + 1:, ch] = im[row, col:, ch]
                output[row, col + 1:, ch] = im[row, col:, ch]

    return output


@jit
def get_minimum_seam(im):
    h, w = im.shape[:2]

    energyfn = backward_energy

    M = energyfn(im)

    print(M.shape)

    backtrack = np.zeros_like(M, dtype=np.int)

    # populate DP matrix
    for i in range(1, h):
        for j in range(0, w):
            if j == 0:
                idx = np.argmin(M[i - 1, j:j + 2])
                backtrack[i, j] = idx + j
                min_energy = M[i-1, idx + j]
            else:
                idx = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]

            M[i, j] += min_energy

    # backtrack to find path
    seam_idx = []
    boolmask = np.ones((h, w), dtype=np.bool)
    j = np.argmin(M[-1])
    for i in range(h-1, -1, -1):
        boolmask[i, j] = False
        seam_idx.append(j)
        j = backtrack[i, j]

    seam_idx.reverse()
    return np.array(seam_idx), boolmask


def seams_removal(im, num_remove, vis=False, rot=False):
    for _ in range(num_remove):
        seam_idx, boolmask = get_minimum_seam(im)
        if vis:
            visualize(im, boolmask, rotate=rot)
        im = remove_seam(im, boolmask)
    return im


def seam_insertion(im, num_add, vis = False, rot = False):
    seams_record = []
    temp_im = im.copy()

    for _ in range(num_add):
        seam_idx, boolmask = get_minimum_seam(temp_im)
        if vis:
            visualize(temp_im, boolmask, rotate=rot)

        seams_record.append(seam_idx)
        temp_im = remove_seam(temp_im, boolmask)

    seams_record.reverse()

    for _ in range(num_add):
        seam = seams_record.pop()
        im = add_seam(im, seam)

        if vis:
            visualize(im, rotate=rot)

        # update the remaining seam indices location in original image (2 seams added, 1 original and 1 duplicate)
        for remaining_seam in seams_record:
            remaining_seam[np.where(remaining_seam >= seam)] += 2

    return im


def seam_carve(im, dy, dx, vis=False):
    im = im.astype(np.float64)
    h, w = im.shape[:2]
    # print(h,w)

    output = im

    if dx < 0:
        output = seams_removal(output, -dx, vis)

    elif dx > 0:
        output = seam_insertion(output, dx, vis)

    if dy < 0:
        output = rotate_image(output, True)
        output = seams_removal(output, -dy, vis, rot=True)
        output = rotate_image(output, False)

    elif dy > 0:
        output = rotate_image(output, True)
        output = seam_insertion(output, dy, vis, rot=True)
        output = rotate_image(output, False)

    return output


if __name__ == '__main__':

    IMAGE = "test.jpeg"
    OUTPUT = "out.jpeg".format(IMAGE.split('.')[0])

    try:
        im = cv2.imread(IMAGE)
        h, w = im.shape[:2]

        if SHOULD_DOWNSIZE:
            if w > DOWNSIZE_WIDTH:
                im = resize(im, width=DOWNSIZE_WIDTH)
                h,w = im.shape[:2]
            if h > DOWNSIZE_HEIGHT:
                im = resize(im, height = DOWNSIZE_HEIGHT)
                h,w = im.shape[:2]

        print("Height = {} and Width = {} \n".format(h, w))

        new_h, new_w = map(int, input("Enter new Height and Width - ").split())

        if new_w > w:
            im = cv2.resize(im, (new_w,h))
            h,w = im.shape[:2]

        if new_h > h:
            im = cv2.resize(im, (w, new_h))
            h,w = im.shape[:2]

        if 0 < new_h <= h and 0 < new_w <= w:
            dy = new_h - h
            dx = new_w - w

            output = seam_carve(im, dy, dx, True)

            cv2.imwrite(OUTPUT, output)
            cv2.waitKey(0)

        else:
            raise ValueError

    except Exception as e:
        print(e)
