import numpy as np
from PIL import Image
from GistConverter import GistConverter
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
class GistDataLoader:
    """
    A class to load the necessary data
    :param gist_extractor: the gist class being used
    """
    def __init__(self, gist_extractor):
        self.gist = gist_extractor

    def load_training_data(self, training_path):
        """
        a function used to load the training data
        """
        X = []
        y = []
        class_names = os.listdir(training_path)
        class_names.pop(0) # gets rid of the .DS_Store file
        for label, class_name in enumerate(class_names): # loops through each label
            class_path = training_path + "/" + class_name
            images = os.listdir(class_path) # gets the list of images
            images.pop(0)
            print(class_name)
            for filename in images: # loops through each image and gets its gist version
                img = Image.open(class_path + "/" + filename).convert('L')
                img = np.array(img)
                gist = self.gist.compute_gist(img)
                X.append(gist)
                y.append(label)

        return np.array(X), np.array(y), class_names

    def load_test_data(self, test_path):
        """
        A function used to load the test data
        """
        filenames = sorted(os.listdir(test_path)) # gets the filenames
        gist_list = []
        for filename in filenames:  # loops through each file
            print(filename)
            img = Image.open(test_path + "/" + filename).convert('L')
            img = np.array(img)
            gist_image = self.gist.compute_gist(img)
            gist_list.append(gist_image) # converts it to its gist version and adds it to a list
        return gist_list, filenames


def main():
    gistExtractor = GistConverter()
    dataLoader = GistDataLoader(gistExtractor)
    train_dir = "../training"
    test_dir = "../testing"

    #this trains the data
    X_train, y_train, class_names = dataLoader.load_training_data(train_dir)

    # this is the model used to train the classifier
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("nb", GaussianNB())
    ])

    model.fit(X_train, y_train)

    # this loads the test data
    X_test, test_filenames = dataLoader.load_test_data(test_dir)

    # this predicts the model based on the test data
    y_pred = model.predict(X_test)

    # this saves the predictions to a text file
    with open("runX.txt.txt", "w") as f:
        for filename, label in zip(test_filenames, y_pred):
            f.write(f"{filename} {class_names[label]}\n")


if __name__ == "__main__":
    main()
