from django.conf import settings
from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format(f"{settings.BASE_DIR}/trmodel", binary=True)
print("MODEL LOADED")
