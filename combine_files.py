import xarray as xr
import fsspec
import logging

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# --- Base config ---
base_path = "s3://ocf-open-data-pvnet/data/uk/pvlive/v1/"
months = [f"{m:02}" for m in range(1, 13)]
year = "2023"
paths = [f"{base_path}target_data_{year}_{month}.zarr" for month in months]

# Create S3 filesystem (anonymous)
fs = fsspec.filesystem("s3", anon=True)

# --- Helper to open Zarr, with fallback to nested store ---
def try_open_zarr(path):
    try:
        mapper = fs.get_mapper(path)
        ds = xr.open_zarr(mapper, consolidated=False)

        if not ds.data_vars:
            nested = f"{path}/{path.split('/')[-1]}"
            logging.warning(f"No variables at top-level. Trying nested path: {nested}")
            nested_mapper = fs.get_mapper(nested)
            ds = xr.open_zarr(nested_mapper, consolidated=True)

        return ds

    except Exception as e:
        logging.error(f"Failed to open Zarr at {path}")
        logging.exception(e)
        return None

# --- Open and collect datasets ---
datasets = []
for path in paths:
    logging.info(f"Attempting to open: {path}")
    ds = try_open_zarr(path)

    if ds is not None:
        logging.info(f"Opened dataset: {path}")
        logging.info(f"Variables: {list(ds.data_vars)}")
        logging.info(f"Coords: {list(ds.coords)}")

        if "datetime_gmt" not in ds.coords:
            logging.warning(f"'datetime_gmt' missing in: {path}")
        else:
            logging.info(f"'datetime_gmt' found")
            datasets.append(ds)

# --- Filter for those with datetime_gmt ---
if not datasets:
    logging.critical("No datasets were loaded. Exiting.")
    exit()

datasets_with_time = [ds for ds in datasets if "datetime_gmt" in ds.coords]

if not datasets_with_time:
    logging.critical("No datasets contain 'datetime_gmt'. Exiting.")
    exit()

# --- Concatenate and save ---
logging.info(f"Concatenating {len(datasets_with_time)} datasets along 'datetime_gmt'...")
try:
    combined = xr.concat(datasets_with_time, dim="datetime_gmt", coords="minimal")

    # Optional: sort and deduplicate datetime_gmt
    combined = combined.sortby("datetime_gmt")
    combined = combined.sel(datetime_gmt=~combined.indexes["datetime_gmt"].duplicated())

    logging.info("Rechunking to ensure Zarr compatibility...")
    combined = combined.chunk({
        'gsp_id': combined.sizes['gsp_id'],     # full slice, unchunked
        'datetime_gmt': 1440                    # chunked along time
    })

    # 🧹 Clean up any old Zarr encoding to prevent write errors
    for var in combined.data_vars:
        if 'chunks' in combined[var].encoding:
            del combined[var].encoding['chunks']

    print(combined.chunks)  # Optional: view final chunking

    combined.to_zarr("combined_2023_gsp.zarr", mode="w", consolidated=True)
    logging.info("✅ Saved combined Zarr to disk successfully.")

except Exception as e:
    logging.error("❌ Concatenation or saving failed")
    logging.exception(e)
