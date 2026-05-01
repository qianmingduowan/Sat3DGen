# Dataset Layout

This release uses **VIGOR** as the only supported dataset.

## Original VIGOR Content

The original VIGOR release provides only the RGB image folders:

- `satellite/`
- `panorama/`

## Additional Content We Provide

We will upload the additional files for this project to one Hugging Face release:

- `sat_depth/`
- `pano_sky_mask/`
- `Seattle_DSM/`
- training split `.txt` files
- test split `.txt` files

Hugging Face placeholder:

- `<HUGGINGFACE_LINK>`

## Expected Layout

The code does **not** include a built-in `data/vigor` directory. Prepare your own VIGOR root like this:

```text
YOUR_VIGOR_ROOT/
|-- train__corrected_all_3city_remove_building.txt
|-- test_remove_building.txt
|-- Seattle/
|   |-- satellite/
|   |-- panorama/
|   |-- pano_sky_mask/
|   `-- sat_depth/
|-- NewYork/
|   |-- satellite/
|   |-- panorama/
|   |-- pano_sky_mask/
|   `-- sat_depth/
|-- SanFrancisco/
|   |-- satellite/
|   |-- panorama/
|   |-- pano_sky_mask/
|   `-- sat_depth/
`-- Seattle_DSM/
```

`Seattle_DSM/` must be placed at the same level as the city folders, not inside `Seattle/`.
