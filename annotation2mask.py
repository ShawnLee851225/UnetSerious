import json
from PIL import Image, ImageDraw

def read_label_json(path):
    with open(path,'r') as f:
        label = json.load(f)
    return label



label_json_path = './dataset/annotations/instances_default.json'
masks_path = './dataset/masks/'

label = read_label_json(label_json_path)
label_key = [key for key in label.keys()]
label_value = [value for value in label.values()]

for j in range(len(label['images'])):
    segmentation = []
    for i,annotations in enumerate(label['annotations']):
        if(label['annotations'][i]['image_id'] == j+1):
            seg = label['annotations'][i]['segmentation'][0]
            segmentation.append(seg)
    
    mask_image = Image.new('L', (1920, 1080), 0)
    draw = ImageDraw.Draw(mask_image)
    for seg in segmentation:
        draw.polygon(seg, fill=255)
    mask_image.save(masks_path+str(label['images'][j]['file_name']))
    # mask_image.show()

