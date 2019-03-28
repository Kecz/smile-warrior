from data_server import load_dataset
from data_augmentation import create_datagen
from train_with_augmentation import prepare_data
import cv2
import glob

path_to_dataset = 'smile_warrior_dataset.csv'
path_to_save_augmented_data = 'augmented_data/augmented'
path_to_save_resized_augmented_data = 'augmented_data/augmented'

choose_train_dataset_from_to = [0, 5]


def main():

    create_augmented(path_to_dataset, path_to_save_augmented_data)
    resize_augmented(path_to_save_resized_augmented_data)


def create_augmented(path_to_dataset, save_directory):
    x_train, y_train, x_test, y_test, x_validate, y_validate, side_length = prepare_data(path_to_dataset)
    datagen = create_datagen()
    datagen.fit(x_train)

    print('Saving augmented_images...')

    i = 0
    for batch in datagen.flow(
            x_train[choose_train_dataset_from_to[0]:choose_train_dataset_from_to[1]],
            batch_size=1,
            save_to_dir=save_directory,
            save_prefix='Augmented',
            save_format='jpeg'):
        i += 1
        if i > 5*16:
            break


def resize_augmented(path_to_augmented):

    list_of_images = []
    for image in glob.glob(path_to_augmented+"/*"):
        list_of_images.append(cv2.imread(image))

    for index, photo in enumerate(list_of_images):
        photo_reshaped = cv2.resize(photo, (600, 600))
        cv2.imwrite('augmented_data/augmented_resized/aug_reshaped'+str(index)+'.png', photo_reshaped)


if __name__ == "__main__":
    main()
