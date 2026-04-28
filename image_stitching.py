import argparse
import glob
import os
import cv2 as cv
import numpy as np


def load_images(pattern):
    if os.path.isdir(pattern):
        exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        filenames = sorted(f for f in os.listdir(pattern) if f.lower().endswith(exts))
        paths = [os.path.join(pattern, f) for f in filenames]
    else:
        paths = sorted(glob.glob(pattern))

    imgs = [cv.imread(p) for p in paths]
    imgs = [im for im in imgs if im is not None]
    return imgs, paths


def create_detector(name='orb'):
    name = name.lower()
    if name == 'sift':
        try:
            return cv.SIFT_create()
        except Exception:
            raise RuntimeError('SIFT not available in this OpenCV build')
    # default ORB
    return cv.ORB_create(5000)


def detect_and_compute(detector, img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    kp, desc = detector.detectAndCompute(gray, None)
    return kp, desc


def match_descriptors(desc1, desc2, detector_name='orb'):
    if desc1 is None or desc2 is None:
        return []
    if detector_name == 'sift':
        # FLANN matcher for SIFT (float descriptors)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv.FlannBasedMatcher(index_params, search_params)
        matches = matcher.knnMatch(desc1, desc2, k=2)
    else:
        # Hamming matcher for ORB (binary descriptors)
        matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
        matches = matcher.knnMatch(desc1, desc2, k=2)

    # ratio test
    good = []
    for m_n in matches:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good


def find_homography(kp1, kp2, matches, reproj_thresh=4.0):
    if len(matches) < 4:
        return None, None
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    H, mask = cv.findHomography(pts2, pts1, cv.RANSAC, reproj_thresh)
    return H, mask


def warp_pair(img_base, img_to_warp, H):
    # compute size of the stitched panorama
    h1, w1 = img_base.shape[:2]
    h2, w2 = img_to_warp.shape[:2]

    corners = np.array([[0,0], [w2,0], [w2,h2], [0,h2]], dtype=np.float32).reshape(-1,1,2)
    warped_corners = cv.perspectiveTransform(corners, H)
    all_corners = np.vstack((warped_corners, np.array([[[0,0]], [[w1,0]], [[w1,h1]], [[0,h1]]], dtype=np.float32)))

    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    translation = [-xmin, -ymin]

    H_trans = np.array([[1,0,translation[0]],[0,1,translation[1]],[0,0,1]])
    out_size = (xmax - xmin, ymax - ymin)

    result = cv.warpPerspective(img_to_warp, H_trans.dot(H), out_size)
    result[translation[1]:translation[1]+h1, translation[0]:translation[0]+w1] = blend_images(
        result[translation[1]:translation[1]+h1, translation[0]:translation[0]+w1], img_base)
    return result


def blend_images(dest_region, src):
    # dest_region and src must have same shape
    h, w = src.shape[:2]
    dest = dest_region.copy()
    if dest.shape[0] != h or dest.shape[1] != w:
        # resize dest region to match src (rare due to rounding)
        dest = cv.resize(dest, (w, h))
    # improved blending using distance transform (feathering) on overlap
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    dest_gray = cv.cvtColor(dest, cv.COLOR_BGR2GRAY)
    mask_src = (src_gray > 0).astype(np.uint8)
    mask_dest = (dest_gray > 0).astype(np.uint8)

    # if no overlap, just place src where dest is empty
    only_src = np.logical_and(mask_src == 1, mask_dest == 0)
    only_dest = np.logical_and(mask_dest == 1, mask_src == 0)
    overlap = np.logical_and(mask_src == 1, mask_dest == 1)

    out = dest.copy()
    out[only_src] = src[only_src]

    if not np.any(overlap):
        return out

    # prepare masks for distance transform (non-zero foreground)
    # distanceTransform requires 8-bit single-channel with non-zero pixels as foreground
    m_src = (mask_src * 255).astype(np.uint8)
    m_dest = (mask_dest * 255).astype(np.uint8)

    # compute distance to the nearest zero (background) inside each mask
    # invert masks so distance grows away from seam inside region
    # but using the mask directly yields distance to background which is fine for feathering
    dist_src = cv.distanceTransform(m_src, cv.DIST_L2, 5).astype(np.float32)
    dist_dest = cv.distanceTransform(m_dest, cv.DIST_L2, 5).astype(np.float32)

    # compute blending weights in overlap: weight_src = dist_src / (dist_src + dist_dest)
    denom = dist_src + dist_dest
    # avoid division by zero
    denom[denom == 0] = 1.0
    weight_src = dist_src / denom
    weight_dest = dist_dest / denom

    # expand to 3 channels
    weight_src_3 = np.repeat(weight_src[:, :, np.newaxis], 3, axis=2)
    weight_dest_3 = np.repeat(weight_dest[:, :, np.newaxis], 3, axis=2)

    # convert to float for blending
    src_f = src.astype(np.float32)
    dest_f = dest.astype(np.float32)

    # apply blending only in overlap region
    blended_overlap = (weight_src_3 * src_f + weight_dest_3 * dest_f).astype(np.uint8)
    out[overlap] = blended_overlap[overlap]
    return out


def stitch_images(images, detector_name='orb'):
    if len(images) == 0:
        return None
    detector = create_detector(detector_name)
    result = images[0].copy()

    for i in range(1, len(images)):
        img = images[i]
        kp1, desc1 = detect_and_compute(detector, result)
        kp2, desc2 = detect_and_compute(detector, img)

        matches = match_descriptors(desc1, desc2, detector_name)
        H, mask = find_homography(kp1, kp2, matches)
        if H is None:
            print(f"Warning: cannot find homography between images 0 and {i}. Skipping.")
            continue
        result = warp_pair(result, img, H)
    return result


def main():
    parser = argparse.ArgumentParser(description='Basic image stitching')
    parser.add_argument('--output', default='panorama.jpg')
    parser.add_argument('--detector', choices=['orb', 'sift'], default='orb')
    args = parser.parse_args()

    imgs, paths = load_images('imgs')
    if len(imgs) < 2:
        print('Need at least two images to stitch')
        return

    pano = stitch_images(imgs, detector_name=args.detector)
    if pano is None:
        print('Stitching failed')
        return

    cv.imwrite(args.output, pano)
    print(f'Panorama saved to {args.output}')


if __name__ == '__main__':
    main()
