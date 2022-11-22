import argparse
import os
import sys

import numpy as np
import shutil
from final_grouping import get_final_clusters



def parse_cli():
    """
    Parses arguments from CLI.
    :return: Source and target directories
    """
    parser = argparse.ArgumentParser()
    # source folder
    parser.add_argument('-s', "--source", help="Enter path to the source folder where all the unsorted photos are") # default="Tel Aviv",
    parser.add_argument('-t', "--target", default="sorted_photos", help="Enter target path to store the sorted photos")

    args = parser.parse_args()
    source_dir = args.source
    target_dir = args.target

    return source_dir, target_dir

def main():
    try:
        source_dir, target_dir = parse_cli()

    except FileNotFoundError as e:
        print('Invalid source path name. Check directory and try again')
        sys.exit()

    # check that source folder is not empty
    if len(os.listdir(source_dir)) == 0:
        print('Source folder has no images.')
        sys.exit()
    # check that source folder has only .jpg images
    elif not all([file.lower().endswith(".jpg"), file.lower().endswith(".jpeg")]for file in os.listdir(source_dir)):
        print('Source folder can only contain .JPG images.')
        sys.exit()
    # check that source folder has at leas 2 images for grouping
    elif len(os.listdir(source_dir)) < 2:
        print('You need at least two photos in the source directory to run the program')
        sys.exit()

    # calls other classes t perform the grouping
    print('[INITIALIZING...]')
    img_paths, final_clusters = get_final_clusters(source_dir)
    
    #check if target folder is in place. If folder doesn't exist, then create it.
    check_target_dir = os.path.isdir(target_dir)
    if not check_target_dir:
        os.makedirs(target_dir)

    # create folders for each of final clusters
    for cluster in np.unique(final_clusters):
        os.mkdir(os.path.join(target_dir, str(cluster)))



    # move all the photos to their respective folders
    for idx, cur_path in enumerate(img_paths):
        move2path = target_dir + '/' + str(final_clusters[idx]) + '/'  # + cur_path.split('/')[-1]
        shutil.move(cur_path, move2path)


if __name__ == '__main__':
    main()
    
    