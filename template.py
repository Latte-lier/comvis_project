import scipy as sc
import cv2 as cv
import numpy as np
import os

def get_all_train_image_labels(path):
    '''
        To get a list of path directories from root path

        Parameters
        ----------
        path : str
            Location of root directory
        
        Returns
        -------
        list
            List containing all train images label
        list
            List containing all train images indexes
    '''
    label_list = os.listdir(path)
    
    index_list = []
    for label in label_list:
        index_list.append(os.listdir(path + label))

    # print(label_list)
    # print(index_list)
    return label_list, index_list

def get_all_train_images(path):
    '''
        Get all Train Images & resize it using the given path

        Parameters
        ----------
        path : str
            Location of root directory
        
        Returns
        -------
        list
            List containing all the resized train images
    '''
    label_list = os.listdir(path)

    resized_img_list = []

    for k, label in enumerate(label_list):
        index_list = os.listdir(path + label)
        for i, index in enumerate(index_list):
            if(i+1) == len(index_list):
                break
            new_path = path + label + '/' + index
            # print(new_path)
            if '.db' in new_path:
                continue
            img = cv.imread(new_path)

            width = 300
            height = int(img.shape[0]/ (img.shape[1]/300))
            dim = (width, height)
            
            resized_img = cv.resize(img, dim, interpolation = cv.INTER_AREA)

            resized_img_list.append((k, resized_img))
            
            # cv.imshow('title',img_gray)
            # cv.waitKey(1000)
    
    # print(resized_img_list)
    return resized_img_list

def detect_faces_and_filter(image_list, image_labels=None):
    '''
        To detect a face from given image list and filter it if the face on
        the given image is not equals to one

        Parameters
        ----------
        image_list : list
            List containing all loaded images
        image_labels : list
            List containing all image classes labels
        
        Returns
        -------
        list
            List containing all filtered and cropped face images in grayscale
        list
            list containing image gray face location
        list
            List containing all filtered image classes label
    '''
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_list = []
    class_list = []
    gray_img = []

    for k, image in image_list:
        img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        
        detected_faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=5)
        if(len(detected_faces) < 1):
            continue
        for face_rect in detected_faces:
            x,y,w,h=face_rect
            face_img = img_gray[y:y+w, x:x+h]
            face_list.append(face_img)
            class_list.append(k)
            gray_img.append(face_rect)
        

    # print(class_list)
    return(face_list, gray_img, class_list)

def train(gray_image_list, gray_labels):
    '''
        To create and train face recognizer object

        Parameters
        ----------
        gray_image_list : list
            List containing all filtered and cropped face images in grayscale
        gray_labels : list
            List containing all filtered image classes label
        
        Returns
        -------
        object
            Classifier object after being trained with cropped face images
    '''
    face_recognizer=cv.face.LBPHFaceRecognizer_create()
    face_recognizer.train(gray_image_list, np.array(gray_labels))

    return face_recognizer

def get_all_test_images(path):
    '''
        To load a list of test images from given path list

        Parameters
        ----------
        path : str
            Location of images root directory
        
        Returns
        -------
        list
            List containing all image in the test directories
    '''
    image_tested = []
    test_image = os.listdir(path)
    for i, filename in enumerate(test_image):
        if(i+1) == len(test_image):
            break
        new_path = path + filename
        # print(new_path)
        if '.db' in new_path:
            continue
        img = cv.imread(new_path)
        image_tested.append((i, img))

    # for iamge in image_tested:
    #     cv.imshow('title', iamge)
    #     cv.waitKey(500)

    return image_tested

def predict(classifier, gray_test_image_list):
    '''
        To predict the test image with the classifier

        Parameters
        ----------
        classifier : object
            Classifier object after being trained with cropped face images
        gray_test_image_list : list
            List containing all filtered and cropped face images in grayscale

        Returns
        -------
        list
            List containing all prediction results from given test faces
    '''

    predicted = []

    for iamge in gray_test_image_list:
        res, _ = classifier.predict(iamge)
        predicted.append(res)

    # print(predicted)
    return predicted

def write_prediction(predict_results, test_image_list, test_faces_rects, train_labels):
    '''
        To draw prediction results on the given test images and resize the image

        Parameters
        ----------
        predict_results : list
            List containing all prediction results from given test faces
        test_image_list : list
            List containing all loaded test images
        test_faces_rects : list
            List containing all filtered faces location saved in rectangle
        train_names : list
            List containing the names of the train sub-directories

        Returns
        -------
        list
            List containing all test images after being drawn with
            final result
    '''

    result = []
    for i, image in test_image_list:
        x,y,w,h = test_faces_rects[i]
        cv.rectangle(image,  (x,y), (x+w, y+h), (254, 204, 51), 2)
        text = train_labels[predict_results[i]]
        cv.putText(image, text, (x,y -10), cv.FONT_HERSHEY_PLAIN, 1.5, (0,255,0), 2)
        # cv.imshow('title', image)
        # cv.waitKey(0)
        result.append(image)
    
    return result
    

def combine_and_show_result(image_list):
    '''
        To show the final image that already combine into one image

        Parameters
        ----------
        image_list : nparray
            Array containing image data
    '''
    final_result = np.hstack((image_list))       

    cv.imshow('Final Result', final_result)
    cv.waitKey(0)

'''
You may modify the code below if it's marked between

-------------------
Modifiable
-------------------

and

-------------------
End of modifiable
-------------------
'''

if __name__ == "__main__":

    '''
        Please modify train_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    current_dir = os.getcwd()
    train_path = str(current_dir + '/dataset/train/')
    '''
        -------------------
        End of modifiable
        -------------------
    '''
    train_image_labels, train_image_indexes = get_all_train_image_labels(train_path)
    train_image_list = get_all_train_images(train_path)
    gray_train_image_list, _, gray_train_labels = detect_faces_and_filter(train_image_list, train_image_indexes)
    
    classifier = train(gray_train_image_list, gray_train_labels)

    '''
        Please modify test_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    test_path = str(current_dir + '/dataset/test/')
    '''
        -------------------
        End of modifiable
        -------------------
    '''

    test_image_list = get_all_test_images(test_path)
    gray_test_image_list, gray_test_location, _ = detect_faces_and_filter(test_image_list)
    predict_results = predict(classifier, gray_test_image_list)
    predicted_test_image_list = write_prediction(predict_results, test_image_list, gray_test_location, train_image_labels)
    
    combine_and_show_result(predicted_test_image_list)