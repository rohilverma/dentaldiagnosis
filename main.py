from utils import read_image
import matplotlib.pyplot as plt

image = read_image("./teeth_xml/JPEGImages/10.png")

plt.imshow(image)
plt.show()


#Take a look at converting xml to csv function


#Load the dataset

from core import Dataset

#If the xml file and images are located in same folder.
#The goal is to convert the xml annotations to csv file and then display the bbox from there.

dataset = Dataset('./teeth_xml/JPEGImages/')



#Visualize

from visualize import show_labeled_image

image, targets = dataset[0]
show_labeled_image(image, targets['boxes'], targets['labels'])

