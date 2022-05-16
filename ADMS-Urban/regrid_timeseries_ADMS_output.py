# Re-gridding the data to a latitude/longitude grid of highest relevant resolution

print(f"Re-gridding run {run}...")
xmin, ymin, xmax, ymax = gpd.points_from_xy(new_ds.longitude.values, 
                                            new_ds.latitude.values).total_bounds
found_one = False
n_cells = None
ref_cell = None
x_coords = None
y_coords = None
NaN_pcent_min = 100
print("Searching for optimal re-gridding parameters...")
for test_n_cells in tqdm(range(300, 1, -1)):
    cell_size = (xmax-xmin)/test_n_cells
    grid_cells = [shapely.geometry.box(x0, y0, x0 - cell_size, y0 + cell_size) 
              for x0 in np.arange(xmin, xmax + cell_size, cell_size) 
              for y0 in np.arange(ymin, ymax + cell_size, cell_size)]
    test_ref_cell = gpd.GeoDataFrame(grid_cells, columns=["geometry"])
    test_x_coords = test_ref_cell.centroid.x.round(12).drop_duplicates()
    test_y_coords = test_ref_cell.centroid.y.round(12).drop_duplicates()
    if len(test_ref_cell) == len(test_x_coords)*len(test_y_coords):
        variable = "NO2"
        i = 0
        # Grid the timeseries data
        cell_list = []
        cell = test_ref_cell.copy()
        class_gdf = gpd.GeoDataFrame(new_ds[variable][i, :].values, 
                         columns=[f"class_{variable}"], 
                         geometry=gpd.points_from_xy(new_ds.longitude.values, new_ds.latitude.values))
        merge = gpd.sjoin(class_gdf, test_ref_cell, how="left", predicate="within")
        dissolve = merge.dissolve(by="index_right", aggfunc="mean")
        cell.loc[dissolve.index, f"class_{variable}"] = dissolve[f"class_{variable}"].values
        cell_list.append(cell[f"class_{variable}"].values.reshape(len(test_x_coords),len(test_y_coords)))
        # Stack the grids into a numpy array
        classes_gridded = np.stack(cell_list, axis=-1)
        NaN_percentage = ((np.sum(np.isnan(classes_gridded)) / (classes_gridded.shape[0] * classes_gridded.shape[1] * classes_gridded.shape[2]))*100)
        if NaN_percentage <= NaN_pcent_threshold:
            found_one = True
            n_cells = test_n_cells
            ref_cell = test_ref_cell
            x_coords = test_x_coords
            y_coords = test_y_coords
            break
        elif NaN_percentage < NaN_pcent_min and not NaN_percentage <= NaN_pcent_threshold:
            NaN_pcent_min = NaN_percentage
            n_cells = test_n_cells
            ref_cell = test_ref_cell
            x_coords = test_x_coords
            y_coords = test_y_coords
if not found_one:
    print(f"Couldn't get data gaps below {NaN_pcent_threshold}%. Minimum achieved was {NaN_pcent_min.round(1)}%.")
    NaN_percentage = NaN_pcent_min

print(f"Selected to re-grid with {n_cells} cells in the x direction, resulting in {NaN_percentage.round(1)}% NaN gaps in the data.")
grid_name = f"gridded_{n_cells}"
variables = [var for var in list(new_ds.data_vars) if "wind" not in var]

