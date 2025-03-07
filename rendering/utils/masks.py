"""
Writing and manipulating 2D image masks.
"""
import cv2
import numpy as np


def to_24bits(seg_code, coarse_mat_code, fine_mat_code):
    """
    Given a segment code, a coarse material code, and a fine material code,
    concatenate them to a 24-bit integer:

        | SEG. CODE | EMPTY | COARSE MAT. CODE | FINE MAT. CODE |
        |   11 bits  |  1 bit  |      5 bits      |     7 bits     |
    """
    return (seg_code << 13) | (coarse_mat_code << 7) | fine_mat_code


def to_24bits_RGB(seg_code, coarse_mat_code, fine_mat_code):
    """
    Given a segment code, a coarse material code and a fine material code,
    get their 24-bit integer, and split it into 3 8-bit integers.
    """
    print("before encoding",seg_code, coarse_mat_code, fine_mat_code)
    code = to_24bits(seg_code, coarse_mat_code, fine_mat_code)
    fine_mat_code = code & 0x7F  # Last 7 bits
    coarse_mat_code = (code >> 7) & 0x1F  # Next 5 bits
    seg_code = (code >> 13) & 0x7FF  # First 11 bits
    print("after decoding",seg_code, coarse_mat_code, fine_mat_code)
    return (code >> 16, (code >> 8) & 0xFF, code & 0xFF)


def remap_colors(img_path, col_map, part_to_idx, part_to_mat_coarse_idx, part_to_mat_fine_idx):
    """
    Remap pixel values based on a color map.
    """
    # Create a old_value => new_value map based on col_map, parts_fine
    col_remap = {}
    for part, pix in col_map.items():
        print(part, pix, part_to_idx[part], part_to_mat_coarse_idx[part], part_to_mat_fine_idx[part])
        col_remap[pix] = to_24bits_RGB(part_to_idx[part],
                                       part_to_mat_coarse_idx[part],
                                       part_to_mat_fine_idx[part])

    # Open the image
    out_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    out_img = out_img.astype(np.uint8)

    # Create target RGB image
    rgb = np.zeros((out_img.shape[0], out_img.shape[1], 3), dtype=np.uint8)

    # Remap the image colors to RGB
    for old_value, new_value in col_remap.items():
        # Count number of indexes where out_img == old_value
        rgb[out_img == old_value] = np.array(new_value)

    return rgb
