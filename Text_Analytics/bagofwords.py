#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

# library
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# create text data
text_data = np.array(['Nhớ nhà quá! Về nhà thôi.',
                      'Ngắm mặt trời mọc',
                      'Tìm nhà nghỉ giá rẻ thôi'])


# Create the bag of words feature matrix
count = CountVectorizer()
bag_of_words = count.fit_transform(text_data)

# Show feature matrix
bag_of_words.toarray()


# Get feature names
feature_names = count.get_feature_names()

# View feature names
feature_names

# Create data frame for bag of words
pd.DataFrame(bag_of_words.toarray(), columns=feature_names)