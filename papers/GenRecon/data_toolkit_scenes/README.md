# Dataset Preparation Toolkit for Scenes

## SAGE10k

### Step 1: Download the data
First, we download the data from Huggingface.
This is quite fast (4s per zip).
```bash
python data_toolkit_scenes/download_data.py --root <DATASET_ROOT> --num 1000 --rank 0 --world_size 2
```
Now, we need to unzip the files.
This is also quite fast.
```bash
python data_toolkit_scenes/unzip.py --root <DATASET_ROOT> --rank 0 --world_size 2
```
If you want to use a validation set, here's the point to split the data into train and validation set.

### Step 2: Build rooms
Now we build the rooms. This is a custom script that creates a .blend file for each room. Note that there is an issue with the door positioning. However, this is the case for the original .glb code export as well.
Quite fast (15s per room).
```bash
python data_toolkit_scenes/build_rooms.py --root <DATASET_ROOT> --rank 0 --world_size 5 --jobs 2
```

### Step 3: Create chunks
Now the rooms are randomly rotated, chunked and normalized.
Should be parallelized (~1 min per room @ 5 chunks).
```bash
python data_toolkit_scenes/create_chunks.py --root <DATASET_ROOT> --crop_size 2.7 3.0 --num_crops 3 --rank 0 --world_size 5 --jobs 2
```

### Step 4a: Create latents
First, we need to create a metadata file for the chunks!
```bash
python data_toolkit_scenes/create_metadata_chunks.py --root <DATASET_ROOT> 
```
To create shape and texture SLats at resolution 1024 and 512 and sparse structure latents at resolution 64:
```bash
python data_toolkit_scenes/get_slats.py --root <DATASET_ROOT> --do_all  --world_size 20 --rank 0
```


### Step 4b: Render images
GPU required but small is sufficient (60s per room @ 1.5 images per m2).
```bash
python data_toolkit_scenes/render_rooms.py --root <DATASET_ROOT> --num_views_per_m2 1.5 --rank 0 --world_size 10
```

### Step 5: Update metadata
Now, we only need to update the metadata.
```bash
python data_toolkit_scenes/build_metadata.py SAGE --root <DATASET_ROOT>
```