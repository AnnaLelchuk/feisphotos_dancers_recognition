import argparse
import os
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
    print('[INITIALIZING...]')
    source_dir, target_dir = parse_cli()
    img_paths, final_clusters = get_final_clusters(source_dir)

    #check if target folder is in place:
    check_target_dir = os.path.isdir(target_dir)

    # If folder doesn't exist, then create it.
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
    
    