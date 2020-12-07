import warnings
import numpy as np
import cv2
from numba import jit
from scipy import ndimage as ndi

import imutils

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


@jit
def backward_energy(im):

    xgrad = ndi.convolve1d(im, np.array([1, 0, -1]), axis=1, mode='wrap')
    ygrad = ndi.convolve1d(im, np.array([1, 0, -1]), axis=0, mode='wrap')

    grad_mag = np.sqrt(np.sum(xgrad ** 2, axis=2) + np.sum(ygrad ** 2, axis=2))

    return grad_mag



@jit
def forward_energy(im):
    """
    Forward energy algorithm as described in "Improved Seam Carving for Video Retargeting"
    by Rubinstein, Shamir, Avidan.

    """
    h, w = im.shape[:2]
    im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)

    energy = np.zeros((h, w))
    m = np.zeros((h, w))

    U = np.roll(im, 1, axis=0)
    L = np.roll(im, 1, axis=1)
    R = np.roll(im, -1, axis=1)

    cU = np.abs(R - L)
    cL = np.abs(U - L) + cU
    cR = np.abs(U - R) + cU

    for i in range(1, h):
        mU = m[i - 1]
        mL = np.roll(mU, 1)
        mR = np.roll(mU, -1)

        mULR = np.array([mU, mL, mR])
        cULR = np.array([cU[i], cL[i], cR[i]])
        mULR += cULR

        argmins = np.argmin(mULR, axis=0)
        m[i] = np.choose(argmins, mULR)
        energy[i] = np.choose(argmins, cULR)

    return energy


@jit
def remove_seam(im, boolmask):
    h, w = im.shape[:2]
    boolmask3c = np.stack([boolmask] * 3, axis=2)
    return im[boolmask3c].reshape((h, w - 1, 3))



@jit
def add_seam(im, seam_idx):

    h, w = im.shape[:2]
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
def get_minimum_seam(im, fn = "Backward"):
    h, w = im.shape[:2]

    if fn == "Backward":
        energyfn = backward_energy
    elif fn == "Forward":
        energyfn = forward_energy

    M = energyfn(im)

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


def seams_removal(im, num_remove, vis=False, rot=False, fn = "Backward"):
    for _ in range(num_remove):
        seam_idx, boolmask = get_minimum_seam(im, fn=fn)
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


def seam_carve(im, dy, dx, vis=False, fn="Backward"):
    im = im.astype(np.float64)
    h, w = im.shape[:2]
    # print(h,w)

    if fn == "Backward":
        M = backward_energy(im)
    elif fn == "Forward":
        M = forward_energy(im)

    content = np.sum(M)

    output = im

    if dx < 0:
        output = seams_removal(output, -dx, vis, fn=fn)

    if dy < 0:
        output = rotate_image(output, True)
        output = seams_removal(output, -dy, vis, rot=True, fn=fn)
        output = rotate_image(output, False)

    if fn == "Backward":
        N = backward_energy(output)
    elif fn == "Forward":
        N = forward_energy(output)

    content2 = np.sum(N)

    percent_content_loss = ((content - content2) / content) * 100

    return output, percent_content_loss


def col_removal(im, num_remove):
    h, w = im.shape[:2]

    M = backward_energy(im)

    temp = []

    for j, i in enumerate(M):
        temp.append((sum(i), j))

    temp.sort()

    col_idx = set([temp[i][1] for i in range(num_remove)])

    output = np.zeros_like(im[:h - num_remove], dtype=np.int)

    ind = 0
    for i, j in enumerate(im):
        if i in col_idx:
            continue
        output[ind] = j
        ind += 1

    return output


def column_carve(im, dy, dx):
    im = im.astype(np.float64)
    h, w = im.shape[:2]

    M = backward_energy(im)

    content = np.sum(M)

    output = im

    if dx < 0:
        output = rotate_image(output, True)
        output = col_removal(output, -dx)
        output = rotate_image(output, False)

    if dy < 0:
        output = col_removal(output, -dy)

    N = backward_energy(output)

    content2 = np.sum(N)

    percent_content_loss = ((content - content2) / content) * 100

    return output, percent_content_loss


def pixels_removal(im, num_remove):
    h, w, d = im.shape

    M = backward_energy(im)

    output = np.zeros(shape=(h, w - num_remove, d), dtype=np.int)

    for ind, row in enumerate(M):
        temp = [(i, j) for j, i in enumerate(row)]
        temp.sort()
        pixels_idx = set([temp[i][1] for i in range(num_remove)])

        k = 0
        for i, j in enumerate(im[ind]):
            if i in pixels_idx:
                continue
            output[ind][k] = j
            k += 1

        # output[ind] = output[ind][:w-num_remove]

    return output


def pixels_carve(im, dy, dx):
    im = im.astype(np.float64)
    h, w = im.shape[:2]

    M = backward_energy(im)

    content = np.sum(M)

    output = im

    if dx < 0:
        output = pixels_removal(output, -dx)

    if dy < 0:
        output = rotate_image(output, True)
        output = pixels_removal(output, -dy)
        output = rotate_image(output, False)

    N = backward_energy(output)

    content2 = np.sum(N)

    percent_content_loss = ((content - content2) / content) * 100

    return output, percent_content_loss


def Max_2D_sliding_window(M, h_, w_, h, w):
    rowSum = [[None] * w] * h
    colSum = [[None] * w] * h

    for i in range(h):
        for j in range(w):
            if j == 0:
                rowSum[i][j] = M[i][j]
            else:
                rowSum[i][j] = M[i][j] + rowSum[i][j - 1]

    for i in range(w):
        for j in range(h):
            if j == 0:
                colSum[j][i] = M[j][i]
            else:
                colSum[j][i] = M[j][i] + colSum[j - 1][i]

    window = 0

    for i in range(h_ - 1):
        for j in range(w_ - 1):
            window += M[i][j]

    maxi = 0

    ans = (None, None)

    start = 0

    for i in range(h_ - 1, h):

        window += rowSum[i][w_ - 1]

        window2 = window
        start2 = 0

        for j in range(w_ - 1, w):

            if start == 0:
                window2 += colSum[i][j]
            else:
                window2 += colSum[i][j] - colSum[start - 1][j]

            if window2 > maxi:
                maxi = window2
                ans = (start, start2)

            if start == 0:
                window2 -= colSum[i][start2]
            else:
                window2 -= colSum[i][start2] - colSum[start - 1][start2]

            start2 += 1

        window -= rowSum[start][w_ - 1]
        start += 1

    return ans


def crop(im, h_, w_, M):
    h, w, d = im.shape

    M_ = [[M[i][j] for j in range(w)] for i in range(h)]

    x, y = Max_2D_sliding_window(M_, h_, w_, h, w)

    output = np.zeros(shape=(h_, w_, d), dtype=np.int)

    for i in range(h_):
        for j in range(w_):
            output[i][j] = im[x + i][y + j]

    return output


def croping(im, new_h, new_w):
    im = im.astype(np.float64)
    h, w = im.shape[:2]

    M = backward_energy(im)

    content = np.sum(M)

    output = im

    output = crop(output, new_h, new_w, M)

    N = backward_energy(output)

    content2 = np.sum(N)

    percent_content_loss = ((content - content2) / content) * 100

    return output, percent_content_loss



def carve_main(image,height,width, algo):

    IMAGE = image
    OUTPUT = "out.jpeg"

    try:
        im = imutils.url_to_image(IMAGE)
        h, w = im.shape[:2]

        if SHOULD_DOWNSIZE and algo in [0,1]:
            if w > DOWNSIZE_WIDTH:
                im = resize(im, width=DOWNSIZE_WIDTH)
                h,w = im.shape[:2]
            if h > DOWNSIZE_HEIGHT:
                im = resize(im, height = DOWNSIZE_HEIGHT)
                h,w = im.shape[:2]

        print("Height = {} and Width = {} \n".format(h, w))

        # new_h, new_w = map(int, input("Enter new Height and Width - ").split())

        new_h = height
        new_w = width

        if new_w > w:
            im = cv2.resize(im, (new_w,h))
            h,w = im.shape[:2]

        if new_h > h:
            im = cv2.resize(im, (w, new_h))
            h,w = im.shape[:2]

        if 0 < new_h <= h and 0 < new_w <= w:
            dy = new_h - h
            dx = new_w - w


            if algo == 0:
                output,_ = seam_carve(im, dy, dx, False, fn = 'Backward')

            elif algo == 1:
                output,_ = seam_carve(im, dy, dx, False, fn = 'Forward')

            elif algo == 2:
                output,_ = column_carve(im, dy, dx)

            elif algo == 3:
                output,_ = pixels_carve(im, dy, dx)

            elif algo == 4:
                output,_ = croping(im, new_h, new_w)


            # cv2.imwrite(OUTPUT, output)
            # cv2.waitKey(0)
            print("Complete")

            is_success, buffer = cv2.imencode(".jpg", output)

            return buffer
        else:
            print("Error in new dimensions")
            raise ValueError

    except Exception as e:
        print(e)

