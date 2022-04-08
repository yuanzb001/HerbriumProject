import json
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
class DataExtraction():
    def __init__(self, indexFileName):
        self.indexFilePath = indexFileName
        self.file = open(self.indexFilePath,'rb')
        #self.file = open()
        self.context = json.load(self.file)

    def getImagePathandLabel(self):
        annotation_df = pd.DataFrame(self.context['annotations'])
        categories_df = pd.DataFrame(self.context['categories'])
        genera_df = pd.DataFrame(self.context['genera'])
        images_df = pd.DataFrame(self.context['images'])
        distances_df = pd.DataFrame(self.context['distances'])
        licenses_df = pd.DataFrame(self.context['license'])
        institutions_df = pd.DataFrame(self.context['institutions'])

        re_df = annotation_df.merge(categories_df, on = 'category_id', how = 'left')
        re_df = re_df.merge(images_df, on = 'image_id', how = 'left')
        return re_df['file_name'].tolist(), re_df['category_id'].tolist()