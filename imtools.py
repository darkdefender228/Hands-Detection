import os

def get_imlist(path):
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]

def split(path, path_test, path_train,for_train=0.8):
    imgs = get_imlist(path)
    count = int(len(imgs) * for_train)
    for index, img in enumerate(imgs):
        if index < count:
            os.rename(img, path_train)
        else:
            os.rename(img, path_test)
    