import os 
import numpy as np
import re

def get_rec_folders(directory):
    # subfolders = [f.path for f in os.scandir(directory) if f.is_dir()]
    #subfolders = [f.path for f in os.scandir(directory) if f.is_dir() and any(char in f.name for char in 'fld')]
    # subfolders = [f.path for f in os.scandir(directory) if f.is_dir() and all(char not in f.name for char in 'fld')]
    subfolders = []
    for root, dirs, _ in os.walk(directory):
        pattern = re.compile(r'^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$')
        subfolders.extend([os.path.join(root, d) for d in dirs if pattern.match(d)])

    return subfolders

import h5py
import numpy as np
import os
import json

def _convert_structured_array_strings(arr):
    """
    The definitive solution to the h5py string conversion error.

    This function rebuilds a structured numpy array by converting any fields
    with fixed-length string dtypes (e.g., '<U64', '<S10') to object dtypes ('O').
    A structured array with object-type strings is unambiguously understood by h5py.

    Args:
        arr (np.ndarray): The original structured array.

    Returns:
        np.ndarray: A new structured array with string fields converted to object type.
    """
    new_dtype_list = []
    for field_name in arr.dtype.names:
        field_dtype = arr.dtype.fields[field_name][0]
        # Check if the field is a fixed-length Unicode or Byte string
        if field_dtype.kind in ['U', 'S']:
            new_dtype_list.append((field_name, 'O'))
        else:
            new_dtype_list.append((field_name, field_dtype))

    # Create a new, empty array with the corrected dtype.
    new_array = np.empty(arr.shape, dtype=new_dtype_list)

    # Copy data from the old array to the new one, field by field.
    # When copying string data into an object field, NumPy correctly
    # creates Python `str` objects, which h5py can handle.
    for name in arr.dtype.names:
        new_array[name] = arr[name]

    return new_array

def save_recording_to_hdf5(file_path, lfp_samples, lfp_timestamps, channel_properties, annotations):
    """
    Saves preprocessed recording data to a structured HDF5 file.

    Version 2.9: Implements the definitive two-part solution for saving structured arrays
    with string fields by pre-converting the data AND explicitly defining the target HDF5 dtype.
    """
    print(f"\n--- Saving data to HDF5 file: {file_path} ---")

    try:
        with h5py.File(file_path, 'w') as f:
            # 1. Store LFP samples
            num_samples, num_channels = lfp_samples.shape
            dset_samples = f.create_dataset(
                'lfp_samples', shape=(num_channels, num_samples), dtype=lfp_samples.dtype,
                chunks=(1, min(1024*16, num_samples)), compression='gzip', shuffle=True
            )
            print(f"Writing 'lfp_samples' dataset with shape {dset_samples.shape}...")
            dset_samples[:] = lfp_samples.T

            # 2. Store timestamps
            print(f"Creating 'lfp_timestamps' dataset with shape {lfp_timestamps.shape}...")
            f.create_dataset('lfp_timestamps', data=lfp_timestamps, compression='gzip')

            # 3. Store recording-wide annotations
            print("Storing recording annotations as root attributes...")
            for key, value in annotations.items():
                try:
                    f.attrs[key] = value
                except TypeError:
                    print(f"  - Serializing complex annotation '{key}' to JSON string.")
                    f.attrs[key] = json.dumps(value, default=str)

            # 4. Store channel properties
            print("Storing channel properties in the 'channel_properties' group...")
            prop_group = f.create_group('channel_properties')
            for key, value in channel_properties.items():
                try:
                    data = np.asarray(value)
                    
                    if data.dtype.names is not None:
                        #print(f"  - Property '{key}' is a structured array. Applying definitive save strategy.")
                        
                        # Part 1: Pre-convert the NumPy data to a compatible format (object strings).
                        converted_data = _convert_structured_array_strings(data)
                        
                        # Part 2: Explicitly define the target HDF5 dtype with variable-length strings.
                        h5_dtype = []
                        for name in data.dtype.names:
                            field_dtype = data.dtype.fields[name][0]
                            if field_dtype.kind in ['U', 'S']:
                                h5_dtype.append((name, h5py.string_dtype(encoding='utf-8')))
                            else:
                                h5_dtype.append((name, field_dtype))
                        
                        # Part 3: Create the dataset with the explicit HDF5 dtype and write the converted data.
                        dset = prop_group.create_dataset(key, shape=data.shape, dtype=h5_dtype)
                        dset[...] = converted_data
                    
                    elif data.dtype.kind in ['S', 'U']:
                        #print(f"  - Property '{key}' is a string array; saving as variable-length strings.")
                        data_as_object_array = data.astype(object)
                        prop_group.create_dataset(key, data=data_as_object_array, dtype=h5py.string_dtype(encoding='utf-8'))

                    elif data.dtype.kind == 'O':
                        #print(f"  - Property '{key}' is a generic object array; serializing to JSON string.")
                        json_string = json.dumps(data.tolist(), default=str)
                        prop_group.create_dataset(key, data=json_string)
                        
                    else:
                        #print(f"  - Property '{key}' is a numerical array; saving with original dtype '{data.dtype}'.")
                        prop_group.create_dataset(key, data=data)
                        
                except Exception as e:
                    print(f"  - CRITICAL: Skipping property '{key}' due to an unexpected error: {e}")

        print(f"\nSuccessfully saved data to {file_path}")

    except Exception as e:
        print(f"\nAn error occurred while writing the HDF5 file: {e}")

class NpxLFPH5Reader:
    """
    A class-based reader for Neuropixels HDF5 files, designed for both
    simple one-line loading and advanced, multi-operation access.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self._file = h5py.File(self.file_path, 'r')
        
        self.lfp_samples = self._file['lfp_samples']
        self.lfp_timestamps = self._file['lfp_timestamps']
        self.channel_properties = self._file['channel_properties']
        
        self.num_channels, self.num_samples = self.lfp_samples.shape
        
        self.annotations = {}
        for key, value in self._file.attrs.items():
            if isinstance(value, str):
                try:
                    self.annotations[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    self.annotations[key] = value
            else:
                self.annotations[key] = value

    @staticmethod
    def load_npx_lfp(file_path, channels=None):
        """
        Loads LFP samples, timestamps, and metadata for specified channels.

        Args:
            file_path (str): The path to the HDF5 file.
            channels (list, slice, or None): The channels to load.

        Returns:
            tuple: (lfp_samples, lfp_timestamps, lfp_metadata)
        """
        with NpxLFPH5Reader(file_path) as reader:
            if channels is None:
                channel_indices = slice(None)
            else:
                channel_indices = channels

            lfp_samples = reader.lfp_samples[channel_indices, :]
            if lfp_samples.ndim == 1:
                lfp_samples = lfp_samples.reshape(1, -1)
            
            num_loaded_channels = lfp_samples.shape[0]
            lfp_timestamps = reader.get_timestamps()

            lfp_metadata = {}

            for key in reader.get_property_keys():
                full_property_data = reader.get_property(key)
                lfp_metadata[key] = full_property_data[channel_indices]
            
            for key in reader.get_annotation_keys():
                lfp_metadata[key] = reader.annotations[key]
            
            return lfp_samples, lfp_timestamps, lfp_metadata

    def get_property_keys(self):
        """Returns a list of all available channel property keys."""
        return list(self.channel_properties.keys())

    def get_annotation_keys(self):
        """Returns a list of all available recording annotation keys."""
        return list(self.annotations.keys())

    def get_timestamps(self):
        """Returns the entire timestamps array."""
        return self.lfp_timestamps[:]

    def get_property(self, property_name):
        """
        Retrieves a channel property, correctly handling different stored types.
        """
        if property_name not in self.channel_properties:
            raise ValueError(f"Property '{property_name}' not found.")

        prop_dataset = self.channel_properties[property_name]

        if prop_dataset.shape == () and h5py.check_string_dtype(prop_dataset.dtype):
            try:
                json_string = prop_dataset[()].decode('utf-8')
                return json.loads(json_string)
            except json.JSONDecodeError:
                return prop_dataset[()]
        else:
            data = prop_dataset[:]
            
            if data.dtype.kind == 'O':
                data = np.char.decode(data.astype('S'), 'utf-8')

            if data.dtype.names is not None:
                for name in data.dtype.names:
                    if data.dtype[name].kind == 'O': 
                        field_data = data[name]
                        data[name] = np.char.decode(field_data.astype('S'), 'utf-8')
            return data

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self._file:
            self._file.close()
            self._file = None

    def __del__(self):
        self.close()

    def __repr__(self):
        return (f"NpxLFPH5Reader(file='{os.path.basename(self.file_path)}', "
                f"channels={self.num_channels}, samples={self.num_samples})")