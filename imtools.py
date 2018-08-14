import os
from sklearn.decomposition import PCA

def get_imlist(path):
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]

def split(path, path_test, path_train,for_train=0.8):
    imgs = get_imlist(path)
    count = int(len(imgs) * for_train)
    for index, img in enumerate(imgs):
        img_name = os.path.split(img)[-1]
        if index < count:
          os.rename(img, os.path.join(path_train, img_name))
        else:
          os.rename(img, os.path.join(path_test, img_name))
            
def reduce_dim(n_comp, img):
    """
    reduce dimension of image using PCA
    """
    pca = PCA(n_components = n_comp)
    pca.fit(img)
    gray_pca = pca.fit_transform(img)
    img_restored = pca.inverse_transform(gray_pca)
    return img_restored