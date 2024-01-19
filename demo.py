import glob
from remover import WatermarksRemover


def get_image_filenames():
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp']
    image_files = []
    for extension in image_extensions:
        image_files.extend(glob.glob(f"data/images/{extension}"))
    return sorted(image_files)


# words to be removed
remove_words = "SIEMENS"
# new fixer instance
ir = WatermarksRemover(words=remove_words, image_save_folder='data/results', show_process=True)
image_names = get_image_filenames()
print('start processing...')
for image_name in image_names:
    # start process
    used_time = ir.repair(image_name)
    print(f'file: {image_name}, used time: {used_time}')
# generate process image
ir.save_process_image()
print('process done.')
