# Book Genre Prediction
Book Genre Prediction using book summaries or blurb with both machine learning and deep learning techniques.

Detectable or valid genres:
* Science Fiction
* Crime Fiction
* Non-fiction
* Children's literature
* Fantasy
* Mystery
* Suspense
* Young adult literature


It uses:
* PyTorch - For deep learning models.
* torchtext v0.9 - For pre-trained word vectors.
* sklearn - For machine learning models.

Dataset:
* CMU dataset.
* goodreads book description for the valida genres.

The best model after testing several is the Support Vector Classification model with 64% of accuracy.

The model can be tested on https://fakebooks.pythonanywhere.com/ which was created using flask framework on pythonanywhere.

More information can be found in the pdf file about the implementation and benchmarks.
