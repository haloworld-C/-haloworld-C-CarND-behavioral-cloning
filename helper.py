#!/usr/bin/python
#!/usr/bin/env python
# define a helper gallerary
import glob
import matplotlib.pyplot as plt
import csv
import pickle
import numpy as np
import cv2
import sklearn
import progressbar
import random
# load data
#@parameter: data file path
#@output: the processed dictary
def load(filepath):
    '''
    Read cameras images to process and nomarlization
    them.
    @return:the images and related labels(steering angle)
    '''
    images = [] # left camera images
    steering_angles = [] # related steering angles

    #read files
    gtFile = open(filepath + 'driving_log.csv') # open index file
    # print(filepath + 'driving_log.csv')
    gtReader = csv.reader(gtFile) # csv parser for annotations file
    next(gtReader)  # skip first img
    # process in loop
    i = 0
    correction = 0.6
    for row in gtReader:
        try:
            if abs(float(row[6])) < 20: 
                center_image = cv2.imread(filepath + row[1])
                images.append(center_image)

                left_image = cv2.imread(filepath + row[3])
                images.append(left_image)

                right_image = cv2.imread(filepath + row[5])
                images.append(right_image)

                steering = float(row[6])
                # if steering == 0:
                #     continue 
                steering_angles.append(steering)
                steering_angles.append(steering + correction)
                steering_angles.append(steering - correction)               
                i += 1
                print('正在处理第'+str(i)+'张图片！', end = '\r')
        except ValueError:
            print(row[6])
    images_augmented = [] #center camera images
    labels = []
    for img, steer in zip(images, steering_angles):
        images_augmented.append(img)
        images_augmented.append(cv2.flip(img,1))
        labels.append(steer)
        labels.append(steer * -1)
    gtFile.close()
    return images_augmented,labels
# preprocess data

def pre_process(train_img):
    '''
    function nomarlization images
    @parameter dataset: the loaded data
    @return p_images: preprocessed images
    '''
    x_input = []
    for x in train_img:
        x_mean = (np.float32(x)-128) / 128
        x_input.append(x_mean)
    
    return x_input
# get samples
# @input csvfile path
# @output images info samples
def get_samples(filepath):
    samples = []
    try:
        with open(filepath + '/driving_log.csv') as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                samples.append(line)
    except:
        print("Open csv file failed!!!")
    return samples
# filter the zero angle pictures
# @para samples: the files get in .csv
# @return filer: return the images' steering angle not equal to zero.
def filter_samples(samples):
    filer = []
    for sample in samples:
        num_r = random.randint(0, 9) # generate a random from 0 to 9
        if float(sample[3]) == 0 and num_r < 6 : # filter when 2/3 percent steering angle is zero 
            pass
        else:
            filer.append(sample)
    return filer
# generate the generator
def generator(samples, filepath, batch_size = 32):
    # samples = get_samples(filepath)
    num_input = len(samples)
    correction = 0.4

    while 1: # loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_input, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                angle = float(batch_sample[3])
    
                for i in range(3):
                    image_path = filepath + '/IMG/'+ batch_sample[i].split('\\')[-1]
                    
                    image = cv2.imread(image_path)
                    images.append(image)
                    images.append(cv2.flip(image,1))
                angles.append(angle)
                angles.append(angle * -1)
                angles.append(angle + correction)
                angles.append((angle + correction) * -1)
                angles.append(angle - correction)
                angles.append((angle - correction) * -1)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)








##TEST
# p = progressbar.ProgressBar()
# Path = './generator_data/generator_data/'

# images, labels = load(Path)

# dataset = {}
# dataset['images'] = images
# dataset['labels'] = labels
# file_w = open('dataset5', 'wb')
# pickle.dump(dataset, file_w)
# file_w.close()
# dataset = pickle.load(open('./dataset3', 'rb'))
# test = np.zeros((160,320,3), dtype = 'float32')
# x_input = pre_process(dataset['images'])
# print(x_input[0])

# plt.imshow(x_input[0])
# plt.show()



