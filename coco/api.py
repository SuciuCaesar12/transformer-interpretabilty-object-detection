from pathlib import Path
from PIL import Image
import json


class CocoPanopticAPI:

    def __init__(self, root: Path, annFile: Path, imgDir: Path, masksDir: Path):
        self.root = root
        self.annFile = annFile
        self.imgDir = imgDir      # relative to root
        self.masksDir = masksDir

        if not self.annFile.exists():
            raise FileNotFoundError(f'Annotation file not found at {self.annFile}.')
        if not self.masksDir.exists():
            raise FileNotFoundError(f'Masks directory not found at {self.masksDir}.')

        print(f'[INFO] loading annotations into memory from {self.annFile.name}...')
        with open(self.annFile, 'r') as f:
            self.annotations = json.load(f)
        print('done!')

        print('creating index...')
        self.categories = self.annotations['categories']
        for cat in self.categories:
            cat['isthing'] = True if cat['isthing'] == 1 else False
        self.images = self.annotations['images']
        self.annotations = self.annotations['annotations']
        self.ids = [img['id'] for img in self.images]
        print('index created!')

    def loadImg(self, image_id):
        return self.loadImgs([image_id])[0]

    def loadAnn(self, image_id):
        return next(filter(lambda ann: ann['image_id'] == image_id, self.annotations), None)

    def loadImgs(self, image_ids):
        images = list(filter(lambda img: img['id'] in image_ids, self.images))
        return [Image.open(self.root / self.imgDir / img['file_name'].replace('\\', '/')) for img in images]

    def getImageIds(self):
        return [img['id'] for img in self.images]

    def getCategoryById(self, category_id):
        return next((cat for cat in self.categories if cat['id'] == category_id), None)
    
    def getPanGT(self, ann) -> Image.Image:
        return Image.open(self.masksDir / ann['file_name'])

    def id2color(self, category_id):
        cat = self.getCategoryById(category_id)
        return cat['color']

    def __getitem__(self, image_id):
        return self.loadImg(image_id), self.loadAnn(image_id)

