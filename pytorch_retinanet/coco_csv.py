import pandas as pd
import json


class COCOCSVGenerator():
	def __init__(self, filename, bucket):
		self.bucket = bucket
		self.js = json.load(open(filename))

	def convert_coco_json_to_csv(self):

		csv = 'class_list.csv'
		class_dict = {}
		with open(csv, 'w') as out:
			# out.write('class_name,id\n')
			for idx, cat in enumerate(self.js['categories']):
				out.write(f"{cat['name']},{idx}\n")
				class_dict[cat['id']] = cat['name']

		csv = 'annotations.csv'
		with open(csv, 'w') as out:
			# out.write('image_path,x1,y1,x2,y2,class_name\n')
			for anno in self.js['annotations']:
				image_name = '0' * (12 - len(str(anno['image_id']))) + str(anno['image_id']) + '.jpg'
				x1 = anno['bbox'][0]
				y1 = anno['bbox'][1]
				x2 = anno['bbox'][0] + anno['bbox'][2]
				y2 = anno['bbox'][1] + anno['bbox'][3]
				class_name = class_dict[anno['category_id']]
				out.write(f"{image_name},{x1},{y1},{x2},{y2},{class_name}\n")

if __name__ == '__main__':
	csv_generator = COCOCSVGenerator('annotations/instances_val2017.json', '')
	csv_generator.convert_coco_json_to_csv()
