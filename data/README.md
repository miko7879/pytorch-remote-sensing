# Airbus Aircrafts Detection Sample Dataset

This dataset is a demonstration version of larger and more advanced deep learning datasets created from Airbus satellite imagery. It is provided for demonstration purpose only.

## Background

[Airbus Defense and Space Intelligence](https://www.intelligence-airbusds.com/) operates the largest commercial satellite constellation combining optical imagery from Pléiades, SPOT, Vision-1 and DMC as well as the radar constellation (consisting of TerraSAR-X, TanDEM-X and PAZ). We are further expanding our sensor capabilities with the upcoming Pléiades Neo constellation providing higher resolution, greater revisits and more acquisition capabilities.

[OneAtlas](https://oneatlas.airbus.com/) provides flexible and easy access to Airbus premium satellite imagery, innovative geospatial analytics, industry-specific insights and more.

## Imagery for training

The `images` folder contains 103 extract of Pleiades imagery at roughly 50 cm resolution. Each each image is stored as a JPEG file of size 2560 x 2560 pixels (i.e. 1280 meters on ground). The locations are various airports worldwide. Some airports appear multiple time at different acquisition dates. Some images also include fog or cloud for diversity.

## Annotations

All aircrafts have been annotated with bounding boxes on the provided imagery. The annotations are provided in the form of closed GeoJSON polygons. A CSV file named `annotations.csv` provides all anotations - one annotation per line with the corresponding filename of the image as `image_id` and the class of the annotation, mainly `Aircraft` or `Truncated_Aircraft` for aircrafts located at the border of the image.

## Extra imagery

A folder named `extras` contains 6 extra images which are not annotated but could be used to test a model on new - unseen before - images. 

## License

This data is licensed under the [**Creative Commons BY-NC-SA 4.0 International**](https://creativecommons.org/licenses/by-nc-sa/4.0/) license: 

You are free to :
- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material

as long as you follow the following terms:
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
- **NonCommercial** — You may not use the material for commercial purposes.
- **ShareAlike** — If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.

## Contact

**We welcome feedback and comments!** This dataset was curated by [jeffaudi](https://twitter.com/jeffaudi) for [Airbus DS Intelligence](https://www.intelligence-airbusds.com/) and annotations provided by [Appen](https://appen.com/). 

Please contact our [sales team](https://www.intelligence-airbusds.com/contact/) for any question related to our satellite imagery offer or to our **OneAtlas** digital services. 