<h1> Automatic photo sorting </h1>
Proof of concept for 
<a href="https://feis.photos">feis.photos</a> to perform automatic photo sorting of dancers during compettions.

Read more: https://medium.com/@annalelchuk/automatic-photos-sorting-8b232f1f745a

The app runs locally from CLI. Tested on a Windows PC.

API:
main.py -s source_dir_path -t target_dir_path
* source_folder_path: directory that stores original unsorted images.
* target_dir_path: directory where you want the grouped photos to be moved to.

Example: python.exe main.py -s "dummy_test_photos" -t "sorted_images"


Dummy photos are included here in case you want to try it yourself.

NOTE!
If VGGFace gives you an error "ModuleNotFoundError: No module named 'keras.engine.topology'", make the following correction in the package:
```
filename = "/usr/local/lib/python3.7/dist-packages/keras_vggface/models.py" # change to  path to python directory that stores keras_vggface files 
text = open(filename).read()
open(filename, "w+").write(text.replace('keras.engine.topology', 'tensorflow.keras.utils'))
```