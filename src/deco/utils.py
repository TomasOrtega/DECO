# src/deco/utils.py
import h5py
import json


def save_results_to_hdf5(group, data):
    """
    Recursively save a dictionary or list to an HDF5 group.
    Handles basic types, NumPy arrays, and structured arrays.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            key = str(key)  # HDF5 keys must be strings
            if isinstance(value, (dict, list)):
                subgroup = group.create_group(key)
                save_results_to_hdf5(subgroup, value)
            elif value is not None:
                try:
                    group.create_dataset(key, data=value)
                except TypeError:
                    # For non-HDF5-native types, save as a JSON string
                    group.create_dataset(key, data=json.dumps(str(value)))
    elif isinstance(data, list):
        # Create a group for the list and numbered items inside it
        for i, item in enumerate(data):
            # Use a generic name for list items
            item_key = str(i)
            if isinstance(item, (dict, list)):
                subgroup = group.create_group(item_key)
                save_results_to_hdf5(subgroup, item)
            elif item is not None:
                try:
                    group.create_dataset(item_key, data=item)
                except TypeError:
                    group.create_dataset(item_key, data=json.dumps(str(item)))


def load_results_from_hdf5(group):
    """
    Recursively load an HDF5 group or dataset into a Python dictionary or list.
    """
    # Base case: If the object is a dataset, read and decode its value.
    if isinstance(group, h5py.Dataset):
        value = group[()]
        # If the value is bytes, attempt to decode it.
        if isinstance(value, bytes):
            try:
                # First, try decoding as a simple UTF-8 string.
                return value.decode("utf-8")
            except UnicodeDecodeError:
                # If that fails, it might be a JSON-encoded object.
                try:
                    return json.loads(value.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    # If all decoding fails, return the raw bytes.
                    return value
        return value

    # If it's a group, determine if it represents a list or a dictionary.
    # A group represents a list if all its keys are strings of digits.
    is_list = False
    if group.keys():
        is_list = all(k.isdigit() for k in group.keys())

    if is_list:
        # Reconstruct the list, ensuring correct order and size.
        max_idx = max(int(k) for k in group.keys())
        list_data = [None] * (max_idx + 1)
        for idx_str, sub_item in group.items():
            list_data[int(idx_str)] = load_results_from_hdf5(sub_item)
        return list_data
    else:
        # Reconstruct the dictionary.
        data = {}
        for key, item in group.items():
            data[key] = load_results_from_hdf5(item)
        return data
