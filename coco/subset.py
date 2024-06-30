'''
Script used to create a subset of the COCO dataset.
'''
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import coco.api as cocoAPI

PATH_ROOT = Path(r'./coco/data/panoptic')
PATH_IMG = Path('val2017')   # relative to PATH_ROOT
PATH_ANNOTATIONS = PATH_ROOT / 'panoptic_annotations_trainval2017/annotations/'
PATH_SAVE = Path(r'./coco/data/panoptic_val_subset')
N_SAMPLES = 10  # number of samples to extract


coco_api = cocoAPI.CocoPanopticAPI(
    root=PATH_ROOT,
    annFile=PATH_ANNOTATIONS / 'panoptic_val2017.json',
    imgDir=PATH_IMG,
    masksDir=PATH_ANNOTATIONS / 'panoptic_val2017'
)


def main():
    image_ids = sorted(coco_api.getImageIds(), key=lambda x: len(coco_api.loadAnn(x)['segments_info']), reverse=True)[:N_SAMPLES]
    
    # create subset directory
    PATH_SAVE.mkdir(parents=True, exist_ok=True)

    with open(PATH_SAVE / 'panoptic_2017.json', 'w') as f:
        json.dump({
            'annotations': coco_api.loadAnns(image_ids), 
            'images': coco_api.getImgInfo(image_ids), 
            'categories': coco_api.categories
        }, f)


if __name__ == '__main__':
    main()