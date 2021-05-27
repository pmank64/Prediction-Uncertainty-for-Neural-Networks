from RoadImage import *
from Dataset import *
import os

class ImgLoader():
    def __init__(self):
        self.images = []
        self.truth = []
        self.pred = []
        self.image_data = []


    def generate(self, horizon, road_width, shoulder, offset, rotation, paint_width, directory, truth_line):
        img = RoadImage(1000, 750, bg_color = (79, 95, 240))
        self.images.append(img)

        truth = img.create(horizon, road_width, shoulder, offset, rotation, paint_width, directory, truth_line)
        self.truth.append(truth)
        
        return truth
    
    def get_annotations(self,  annotations_file):
        data = {'name':[], 'truth':[]}
        for i, image in enumerate(self.images):
            data['name'].append(image.file_name)
            data['truth'].append(self.truth[i])
        pd.DataFrame.from_dict(data).to_csv(annotations_file, header=False, index = False)
        return pd.DataFrame.from_dict(data)
    
    def getLoaders(self, annotations_file, img_dir,transform, target_transform):
        annotations_pd = self.get_annotations(annotations_file)
        dataset = Dataset(annotations_file, img_dir, transform, target_transform)
        return (dataset, annotations_pd)

    def save_file(self, img_index, path, format = 'png'):
        self.images[img_index].getIMGObj().save(path, format=format)
        return self.images[img_index]

    def getTruth(self):
        return self.truth

    def getPred(self):
        return self.pred
    
    def set_predictions(self, pred):
        self.pred = pred

    def getImgData(self):
        return self.image_data

    def save_truth_to_disk(self, data, path):
        np.save(path + "truth.npy", np.array(data))
    
    def save_truth_data(self, data):
        self.truth = data

    def save_img_data(self, data):
        self.image_data.append(data)

    def img_data_to_disk(self, i, path):
        np.save(path + "road" + str(i) + ".npy", self.image_data[i])
        
    def load_img_data_from_file(self, path):
        for file in os.listdir(path):
            if file.endswith('.npy') & file.startswith('road'):
                self.image_data.append(np.load(path + file, allow_pickle=True))

    def load_truth_data_from_file(self, path):
        self.truth = np.load(path + 'truth.npy', allow_pickle=True)

    def load_pred_data_from_file(self, path):
        self.truth = np.load(path + 'pred_y.npy', allow_pickle=True)

    def get_by_filename(self, file_name):
        for image in self.images:
            if image.file_name == file_name:
                return image.img