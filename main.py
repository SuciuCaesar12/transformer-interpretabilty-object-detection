from xai_detr.explainer import DetrExplainer
from pathlib import Path
from PIL import Image

import transformers as tr
import argparse
import yaml


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_config', type=str, required=True)
    parser.add_argument('--path_ckpt', type=str, required=True)
    parser.add_argument('--path_images', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--path_output', type=str, required=True)
    
    return parser.parse_args()


def recursive_eval(d: dict):
    for k, v in d.items():
        if isinstance(v, dict):
            recursive_eval(v)
        else:
            try:
                if not v == 'all':
                    d[k] = eval(v)
            except (SyntaxError, NameError, TypeError):
                pass
    return d


def load_model(path_ckpt: str):
    return tr.DetrForObjectDetection.from_pretrained(path_ckpt)


def main():
    args = read_args()
    
    with open('config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = recursive_eval(config)
    
    model = load_model(args.path_ckpt)
    processor = tr.DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    
    detr_explainer = DetrExplainer(
        model=model,
        processor=processor,
        id2label=config['id2label'],
        no_object_id=config['no_object_id'],
        device=args.device
    )
    
    for img_path in Path(args.path_images).rglob('*'):
        image = Image.open(img_path)
        
        detr_explainer.explain(
            image=image,
            include_labels=config['include_labels'],
            threshold=config['threshold'],
            output_dir=Path(args.path_output) / img_path.stem
        )


if __name__ == '__main__':
    main()
