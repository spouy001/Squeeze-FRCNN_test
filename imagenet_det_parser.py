import os
from xml.etree import ElementTree


def retrieve_annotation(image_folder, annotation_folder, file_list, image_set, concept_count, concept_mapping):
    # Obtain annotation files and update the data set
    for this_file in file_list:
        this_image_file = os.path.join(image_folder, this_file)
        this_annotate_file = os.path.join(annotation_folder, this_file.replace('JPEG', 'xml'))
        if not os.path.isfile(this_annotate_file):
            raise Exception('Error: Cannot find annotation file for ' + this_annotate_file)

        # Parse annotation file in XML format
        et = ElementTree.parse(this_annotate_file)
        element = et.getroot()

        element_objects = element.findall('object')
        element_width = int(element.find('size').find('width').text)
        element_height = int(element.find('size').find('height').text)

        # Ignore images without concepts
        if len(element_objects) == 0:
            continue

        annotation_data = {'filepath': this_image_file, 'width': element_width,
                           'height': element_height, 'bboxes': []}
        for element_object in element_objects:
            concept_name = element_object.find('name').text
            if concept_name not in concept_count:
                concept_count[concept_name] = 1
            else:
                concept_count[concept_name] += 1

            if concept_name not in concept_mapping:
                concept_mapping[concept_name] = len(concept_mapping)

            object_bndbox = element_object.find('bndbox')
            xmin = int(round(float(object_bndbox.find('xmin').text)))
            ymin = int(round(float(object_bndbox.find('ymin').text)))
            xmax = int(round(float(object_bndbox.find('xmax').text)))
            ymax = int(round(float(object_bndbox.find('ymax').text)))
            this_annotation = {'class': concept_name, 'x1': xmin, 'x2': xmax, 'y1': ymin, 'y2': ymax}
            annotation_data['bboxes'].append(this_annotation)
        image_set.append(annotation_data)


def get_training_images(training_sample_path):
    # Store information of images in training set
    #train_images = []
    # Store information of images in validation set
    valid_images = []
    # Count the number of training sample in each concept
    concept_count = {}
    # Assigned Index for each concept
    concept_mapping = {}

    # Obtain the paths of all the images (assume that images locate in the folders under the given path)
    #print('Retrieving images for training...')
    #train_image_folder = training_sample_path + 'ILSVRC2014_DET_train'
    #train_image_files = []
    #for folder in os.listdir(train_image_folder):
    #    this_folder = os.path.join(train_image_folder, folder)
    #    if os.path.isdir(this_folder):
    #        for filename in os.listdir(this_folder):
    #            if filename.endswith('.JPEG'):
    #                train_image_files.append(os.path.join(folder, filename))

    #print(str(len(train_image_files)) + ' training images found.')

    # Obtain the paths of all the images (assume that images directly locate in the given path)
    print('Retrieving images for validation...')
    valid_image_folder = training_sample_path + 'ILSVRC2013_DET_val'
    valid_image_files = []
    for filename in os.listdir(valid_image_folder):
        if filename.endswith('.JPEG'):
            valid_image_files.append(os.path.join(filename))

    print(str(len(valid_image_files)) + ' validation images found.')



    # Obtain Annotations for validation images
    print('Retrieving annotations of validation images')
    valid_annotate_folder = training_sample_path + 'ILSVRC2013_DET_bbox_val'
    retrieve_annotation(image_folder=valid_image_folder, annotation_folder=valid_annotate_folder,
                        file_list=valid_image_files, image_set=valid_images,
                        concept_count=concept_count, concept_mapping=concept_mapping)

    return valid_images, concept_count, concept_mapping
