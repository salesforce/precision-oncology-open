import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import pickle
import os
import cv2
from PIL import Image
import argparse

# Example command:
# python3 quilt_generation.py --tissue_label_dir /export/medical_ai/ucsf/tissue_vs_non_pkl/RTOG-9413/ \
#     --tissue_dir /export/medical_ai/ucsf/RTOG-9413/tissue_pickles/ \
#     --feature_dir /export/medical_ai/ucsf/RTOG-9413/tissue_features_simclr/ \
#     --slide_df /export/medical_ai/kaggle_panda/RTOG_pandanet_labels/RTOG-9413.csv \
#     --save_dir /test/ \
#     --save_features

parser = argparse.ArgumentParser(description='Create tissue quilts for all slides of a single patient')
parser.add_argument('--tissue_label_dir', type=str, required=True)
parser.add_argument('--tissue_dir', type=str, required=True)
parser.add_argument('--feature_dir', type=str, required=True)
parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument('--slide_df_loc', type=str, required=True)
parser.add_argument('--save_features', action='store_true', default=False)
parser.add_argument('--save_images', action='store_true', default=False)
parser.add_argument('--num', type=int, default=-1, help="number of samples to process, all if set to -1")
parser.add_argument('--feature_dim', type=int, default=128, help="feature dimension for feature blocks.")
args = parser.parse_args()

class rect_obj():
    """Rect object, containing the size of a rectangle around a piece of tissue. 
    It also includes which patch in the original image should be in which location. 
    """
    def __init__(self, rect_id, rect_df, contour):
        self.rect_id = rect_id
        self.rect_df = rect_df
        self.contour = contour
        x_coords = [x[0] for x in contour]
        y_coords = [x[1] for x in contour]
        self.x = min(x_coords)
        self.y = min(y_coords)
        self.sq_x = max(x_coords) - min(x_coords) + 1
        self.sq_y = max(y_coords) - min(y_coords) + 1
        self.canvas_x = None
        self.canvas_y = None
        self.placed=False
        
        points_in_contour = []
        for x, y in zip(list(rect_df.x), list(rect_df.y)):
            points_in_contour += [(x,y) in contour]
        self.rect_df["poly_match"] = points_in_contour
        self.rect_df = self.rect_df.loc[self.rect_df.poly_match != False]


def load_df(tissue_label_dir, tissue_dir, feature_dir, save_images, save_features, tissue_pickle):
    """Load slide dataframe and add images and feature vectors

    Args:
        tissue_label_dir (String): folder containing labels on which patches contain tissue
        tissue_dir (String): folder containing tissue pickles per slide
        feature_dir (String): folder containing features for above tissue pickles
        save_images (Bool): whether the image quilts should be saved
        save_features (Bool): whether the feature quilts should be saved
        tissue_pickle (String): Tissue_pickle name of slide we're loading the df for

    Returns:
        Pandas df: df for a slide containing images and features
    """
    with open(tissue_label_dir + "/" + tissue_pickle, "rb") as f:
        df = pickle.load(f)
    if save_images:
        with open(tissue_dir + "/" + tissue_pickle, "rb") as f:
            imgs = list(pickle.load(f))
    if save_features:
        with open(feature_dir + "/" + tissue_pickle, "rb") as f:
            features = list(pickle.load(f))
    df = df.loc[df.tissue_vs_non == True]
    df["x"] = df.path.apply(lambda x: int(x.split("_")[-2]))
    df["y"] = df.path.apply(lambda x: int(x.split("_")[-3]))
    if save_features:
        df["features"] = features
    if save_images:
        df["imgs"] = imgs
    
    return df

def get_neighbours(x, y, thresh_shape):
    """Get direct neighbours for an x, y location in a square

    Args:
        x (int): x coord
        y (int): y coord
        thresh_shape ((int, int)): tuple containing square shape

    Returns:
        List: list of tuples with x,y coords
    """
    x_lim, y_lim = thresh_shape
    neighbours = []
    if x > 0:
        neighbours += [(x-1, y)]
    if y > 0:
        neighbours += [(x, y-1)]
    if y < (y_lim - 1):
        neighbours += [(x, y+1)]
    if x < (x_lim - 1):
        neighbours += [(x+1, y)]
    return neighbours

def find_contours(thresh, df):
    """Given a binary image, find all contours 

    Args:
        thresh (np array): array containing binary image
        df (Pandas df): Df containing coordinates of all positive locations in the binary image

    Returns:
        List of lists containing tuples: Every contour is a list containing all 
        coordinates that belong to it. 
    """
    contours = []
    to_process = [(x,y) for x,y in zip(df.x, df.y)]
    processed = []
    
    for x,y in to_process:
        if (x,y) in processed:
            continue
        
        new_contour = [(x,y)]
        contour_processor = get_neighbours(x,y, thresh.shape)
        while contour_processor != []:
            x,y = contour_processor[0]
            contour_processor = contour_processor[1:]
            if thresh[x,y]==255 and (x,y) not in new_contour:
                new_contour += [(x,y)]
                processed += [(x,y)]
                contour_processor += get_neighbours(x,y, thresh.shape)
        contours += [new_contour]
    return contours

def get_slide_matrix(df):
    """Create a binary image where tissue = white, no tissue = black

    Args:
        df (pandas df): dataframe containing coordinates of all tissue patches

    Returns:
        slide_matrix (np.array): array containing binary images
        contours (list of list with tuples): List of lists containing tuples: Every contour is a 
        list containing all coordinates that belong to it. 
    """
    slide_matrix = np.zeros((max(list(df.x)) + 1, max(list(df.y)) + 1),dtype=np.uint8)
    for x, y in zip(list(df.x), list(df.y)):
        slide_matrix[x][y] = 255
    ret, thresh = cv2.threshold(slide_matrix, 127, 256, 0)
    contours = find_contours(thresh, df)
    return slide_matrix, contours

def spiral(X, Y):
    """Create a spiral from the center of a square of size X by Y

    Args:
        X (Int): X coord
        Y (Int): Y coord

    Returns:
        List of tuples: coordinates of spiral, starting from the center
    """
    x = y = 0
    dx = 0
    dy = -1
    coords = []
    for i in range(max(X, Y)**2):
        if (-X/2 < x <= X/2) and (-Y/2 < y <= Y/2):
            coords += [(x,y)]
        if x == y or (x < 0 and x == -y) or (x > 0 and x == 1-y):
            dx, dy = -dy, dx
        x, y = x+dx, y+dy
    return coords

def fill_canvas_top_left(rects, canvas_x=40, canvas_y=30):
    """Place rectangle objects in an image starting at the top left

    Args:
        rects (List of rectangle objects): Rectangles to be placed
        canvas_x (int, optional): canvas width. Defaults to 40.
        canvas_y (int, optional): canvas height. Defaults to 30.

    Returns:
        np.array: Array where rectangles occupy space and the value of a location
        corresponds to a rectangle id (or 0 if no rectangle occupies that space).
    """
    canvas = np.zeros((canvas_x,canvas_y))
    rect_gen = (x for x in rects)
    
    placed = False
    num_placed = 0
    curr_rect = next(rect_gen)
    while True:
        placed = False
        for y in range(0, canvas_y, 1):
            for x in range(0, canvas_x, 1):
                if ((x + curr_rect.sq_x) > canvas_x ) or ((y + curr_rect.sq_y) > canvas_y):
                    continue
                candidate = canvas[x:x+curr_rect.sq_x, y:y+curr_rect.sq_y]
                if not candidate.any():
                    canvas[x:x+curr_rect.sq_x, y:y+curr_rect.sq_y] = curr_rect.rect_id
                    curr_rect.canvas_x = x
                    curr_rect.canvas_y = y
                    curr_rect.placed = True
                    num_placed += 1
                    try:
                        curr_rect = next(rect_gen)
                    except StopIteration:
                        return canvas, True
                    placed = True
        if placed == False:
#             print("Could not find a spot for an object. Placed {} out of {} rectangles".format(num_placed, len(rects)))
            return canvas, False
            
    return canvas

def fill_canvas_center(rects, canvas_x=40, canvas_y=40, offset=10):
    """Place rectangles starting from the center of the canvas

    Args:
        rects (List of rectangle objects): Rectangles to be placed
        canvas_x (int, optional): canvas width. Defaults to 40.
        canvas_y (int, optional): canvas height. Defaults to 30.
        offset (int, optional): As rectangles are placed towards the bottom right,
        we add an offset such that the first rectangle is more centered.. Defaults to 10.

    Returns:
        np.array: Array where rectangles occupy space and the value of a location
        corresponds to a rectangle id (or 0 if no rectangle occupies that space).
    """
    canvas = np.zeros((canvas_x,canvas_y))
    rect_gen = (x for x in rects)
    placement_coords = spiral(canvas_x + offset, canvas_y+offset)
    m_center = np.array((int(canvas_x/2)-offset, int(canvas_y/2)-offset))
    
    placed = False
    num_placed = 0
    curr_rect = next(rect_gen)
    while True:
        placed = False
        for coord in placement_coords:
            x = coord[0] + m_center[0]
            y = coord[1] + m_center[1]
            if min([x,y]) < 0 or (x + curr_rect.sq_x) > canvas_x or (y + curr_rect.sq_y) > canvas_y:
                continue
            candidate = canvas[x:x+curr_rect.sq_x, y:y+curr_rect.sq_y]
            if not candidate.any():
                canvas[x:x+curr_rect.sq_x, y:y+curr_rect.sq_y] = curr_rect.rect_id
                curr_rect.canvas_x = x
                curr_rect.canvas_y = y
                curr_rect.placed = True
                num_placed += 1
                try:
                    curr_rect = next(rect_gen)
                except StopIteration:
                    return canvas, True
                placed = True
                break
        if placed == False:
#             print("Could not find a spot for an object. Placed {} out of {} rectangles".format(num_placed, len(rects)))
            return canvas, False                    
    return canvas

def create_thumbnail(rects, size, resize_res=32):
    """Place tissue patches in a large image, as described in the canvas

    Args:
        rects (Rectangle objects): rectangle objects that have been placed in a canvas
        size (Tuple of ints): shape of canvas
        resize_res (int, optional): reshaped size of tissue patches. Defaults to 32.

    Returns:
        np.array: Array of size shape * image resolution
    """
    empty_img = np.ones((size*resize_res, size*resize_res, 3))*255
    for rect in rects:
        for i, row in rect.rect_df.iterrows():
            if rect.placed:
                img = np.array(Image.fromarray(row.imgs).resize((resize_res,resize_res)))
                relative_x = row.x - rect.x
                relative_y = row.y - rect.y
                new_x = relative_x + rect.canvas_x
                new_y = relative_y + rect.canvas_y
                empty_img[new_x*resize_res:new_x*resize_res+resize_res, new_y*resize_res:new_y*resize_res+resize_res] = img
    return empty_img

def create_feature_block(rects, size, dim=128):
    """Place feature vectors in the block according to how the rectangles have
    been placed in the canvas

    Args:
        rects (List of rect objects): rectangles placed in canvas
        size (tuple of ints): Canvas size
        dim (int, optional): Number of dimensions that the feature vectors have. Defaults to 128.

    Returns:
        np.array: feature block 
    """
    feature_block = np.zeros((size, size, dim))
    for rect in rects:
        for i, row in rect.rect_df.iterrows():
            if rect.placed:
                features = row.features
                relative_x = row.x - rect.x
                relative_y = row.y - rect.y
                new_x = relative_x + rect.canvas_x
                new_y = relative_y + rect.canvas_y
                feature_block[new_x, new_y, :] = features
    return feature_block


def save_slide_thumbnail(tissue_pickle, save_dir, save_features=False, save_images=False):
    """Create the thumbnail for a single slide

    Args:
        tissue_pickle (String): tissue pickle filename
        save_dir (String): path to save directory
        save_features (bool, optional): Save feature block. Defaults to False.
        save_images (bool, optional): Save image quilt. Defaults to False.
    """
    # todo: pass as arguments
    df = load_df(tissue_label_dir, tissue_dir, feature_dir, load_features, tissue_pickle)
    slide_matrix, contours = get_slide_matrix(df)

    rects = []
    sum_in_cont = 0
    for i, contour in enumerate(contours):
        rects += [rect_obj(i+1, df, contour)]
        sum_in_cont += len(rects[-1].rect_df)

    assert len(df) == sum_in_cont, "not all tiles were matched to a contour"

    size = 200
    while True:
        c, succes = fill_canvas_center(rects, size,size)
        if succes:
            break
        else:
            size += 5
    
    if save_images:
        img = create_thumbnail(rects, size)
        img = Image.fromarray(img.astype(np.uint8)).resize((2000,2000))
        img.save(save_dir + "/" + tissue_pickle[:-4] + "_quilt.tiff")
    if save_features:
        feature_block = create_feature_block(rects, size)
        with open(save_dir + "/" + tissue_pickle[:-4] + "_quilt_feature.pkl", "wb") as f:
            pickle.dump(feature_block, f)

def save_patient_thumbnail(slide_df, save_dir, tissue_dir, feature_dir, tissue_label_dir, feature_dim, num=10, save_features=False, save_images=False):
    """Collect all tissue for a single patient and generate a quilt

    Args:
        slide_df (Pandas df): df containing slide information
        save_dir (String): save directory
        tissue_dir (String): directory containing tissue pickles
        feature_dir (String): directory containing feature vectors
        tissue_label_dir (String): directory containing tissue patch labels
        num (int, optional): Number of patients to process. Defaults to 10.
        save_features (bool, optional): Save feature block. Defaults to False.
        save_images (bool, optional): Save image quilt. Defaults to False.
    """
    for patient in tqdm.tqdm(slide_df.cn_deidentified.unique()[:num]):
        patient_df = slide_df.loc[slide_df.cn_deidentified == patient]
        rects = []
        for image_id in list(patient_df.image_id):
            if not os.path.exists(tissue_dir + "/" + image_id.split("/")[-1]):
                print("skipped because tissue_pkl does not exist: ", image_id.split("/")[-1])
                continue
            df = load_df(tissue_label_dir, tissue_dir, feature_dir, save_images, save_features, image_id.split("/")[-1])
            try:
                slide_matrix, contours = get_slide_matrix(df)
            except: 
                print(df)
            sum_in_cont = 0
            for i, contour in enumerate(contours):
                rects += [rect_obj(i+1, df, contour)]
                sum_in_cont += len(rects[-1].rect_df)
            assert len(df) <= sum_in_cont, "not all tiles were matched to a contour"
#         print(len(df), sum_in_cont)
        if len(rects) == 0:
            print("patient {} has no tissue".format(patient))
            continue

        c, succes = fill_canvas_center(rects, 200,200)
        if not succes:
            print("patient {} has too much tissue to fit in 200 by 200. skipping".format(patient))
        if save_images:
            img = create_thumbnail(rects, 200)
            img = Image.fromarray(img.astype(np.uint8)).resize((2000,2000))
            img.save(save_dir + "/" + str(patient) + "_quilt.tiff")
        if save_features:
            feature_block = create_feature_block(rects, 200, feature_dim)
            with open(save_dir + "/" + str(patient) + "_quilt_feature.pkl", "wb") as f:
                pickle.dump(feature_block, f)


tissue_label_dir = args.tissue_label_dir 
tissue_dir = args.tissue_dir 
feature_dir = args.feature_dir 
slide_df = pd.read_csv(args.slide_df_loc) 
save_dir = args.save_dir 
save_features = args.save_features
save_images = args.save_images
feature_dim = args.feature_dim
if args.num == -1:
    num = len(slide_df.cn_deidentified.unique())
else:
    num = args.num
save_patient_thumbnail(slide_df, save_dir, tissue_dir, feature_dir, tissue_label_dir, feature_dim, num=num, save_features=save_features, save_images=save_images)


