from RoadImage import *
from Dataset import *
import os

class ImgLoader(): 
    def __init__(self):
        self.images = []
        self.truth = []
        self.pred = []
        self.image_data = []
        # self.mu = 0
        # self.sigma = 1000

    def reset(self):
        self.images = []
        self.truth = []
        self.pred = []
        self.image_data = []

    def generate(self, horizon, road_width, shoulder, offset, rotation, paint_width, directory, truth_line):
        img = RoadImage(500, 375, bg_color = (79, 95, 240))
        self.images.append(img)
        truth = img.create(horizon, road_width, shoulder, offset, rotation, paint_width, directory, truth_line)
        
        # generating random truth values
        # t1 = np.random.uniform(0, 500, 1)
        # t2 = np.random.uniform(0, 500, 1)
        # self.truth.append([float(t1), float(t2)])

        # self.truth.append(truth)

        return truth

    def make_noise(self, sigma_set):
        # iterate through the RoadImage objects, pulling out the pillow image object
        sigmas = []
        count = 0
        for i, image in enumerate(self.images):

            # normally distributed noise
            # generate mean and standard deviation
            # mu = np.random.normal(0, 10, 1)
            # sigma = np.random.normal(0, 10, 1)
            sigma = sigma_set
            sigma = np.random.uniform(0,sigma,1)
            sigma = sigma[0]
            # sigma = 10
            noisy_func = lambda x: x + ((np.random.normal(0, sigma, len(x)) * np.sqrt(x)) )
            # print("SIGMA: " + str(sigma[0]))
            sigmas.append(sigma)
            pixel_array = np.asarray(image.img, 'float32')
            # image.img.show()
            
            # must transpose indices to make RGB dim first
            pixel_array = pixel_array.transpose(2,0,1)
            print(len(pixel_array))
            # iterate through the color channels and add noise
            channels = []
            for c_channel in pixel_array:
                channels.append([noisy_func(xi) for xi in c_channel])
            noisy_tensor = torch.tensor(channels)
            print(noisy_tensor)
            noisy_tensor = torch.clamp(noisy_tensor, min=0, max=255)
            print(noisy_tensor)
            demo_array = np.moveaxis(noisy_tensor.numpy(), 0, -1)
            demo_array = demo_array.transpose(2,0,1)
            demo_array = torch.tensor(demo_array).numpy()
 
            self.truth.append([float(sigma)])

            self.image_data.append(demo_array)

            count = i
        
        print("average sigmas: " + str(np.sum(np.array(sigmas))/count + 1))
        print("SD sigmas: " + str(np.std(sigmas)))

    
    def get_annotations(self,  annotations_file):
        data = {'name':[], 'truth':[]}
        for i, image in enumerate(self.images):
            data['name'].append(image.file_name)
            data['truth'].append(self.truth[i])
        pd.DataFrame.from_dict(data).to_csv(annotations_file, header=False, index = False)
        return pd.DataFrame.from_dict(data)
    
    def getLoaders(self, annotations_file, img_dir,transform, target_transform):
        annotations_pd = self.get_annotations(annotations_file)
        dataset = Dataset(annotations_file, img_dir, transform, target_transform, self.image_data)
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