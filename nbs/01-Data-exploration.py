from IPython.core.display import display, HTML
display(HTML("<style>.container { width:98% !important; }</style>"))

# ## Imports

# %matplotlib inline

import os
import cv2
import pandas as pd
import hvplot.pandas
import matplotlib.pyplot as plt

data_path = '/home/henrique/Projects/jaguar/datasets/'

# ## Load and check labels

labels = pd.read_csv(os.path.join(data_path, 'trainLabels.csv'))
labels.head(10)

labels[['patient', 'side']] = labels.apply(lambda x: x['image'].split('_'), axis=1, result_type='expand')
labels.patient = labels.patient.astype(int)

labels.head(10)

len(labels), labels.image.nunique()

# No duplicated images

labels.level.value_counts().hvplot.bar(grid=True)


# Dataset seems a bit unbalanced, let's now check if there are discrepancies on the levels on each eye for each patient

def sub(values):
    return values.diff().dropna().abs()


patients = labels.groupby(by='patient')['level'].agg(sub).to_frame(name='level_difference')
patients.head()

# Confirm sizes

labels.patient.nunique(), len(patients)

patients.level_difference.value_counts()

patients.level_difference.value_counts().hvplot.bar(grid=True)

# In most of the cases for each patient the assigned level is consistent between both eyes

# ## Load and check images

# Let's have a look of the raw images. 5 images for each one of the levels

fig = plt.figure(figsize=(25, 16))
for lvl in sorted(labels.level.unique()):
    sampled_imgs = labels[labels.level == lvl]['image'].sample(5, random_state=42).values
    for k, img in enumerate(sampled_imgs):
        im = cv2.imread(os.path.join(data_path, 'resized_train', f'{img}.jpeg'))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        
        ax = fig.add_subplot(5, 5, lvl * 5 + k + 1, xticks=[], yticks=[])
        ax.set_title(f'{lvl}-{img}-{im.shape}')
        plt.imshow(im)

# Each row shows a severity level and at first glance some problems arise:
# * Differences on illumination. Some images are too dark compared to others.
# * Some images have a more uninformative areas (black areas). This can be a problem when reducing the image size because the useful areas become too small. Also, images have different sizes, so maybe we should start by trying to crop images to their infomative dimension. 


