import os

def check_flag_status(flag_file_path):
    """ 
    Check the flag file to see if the process has been run before.
    If the flag file doesn't exist, assume it's the first run.
    
    Returns:
    - bool: True if the process is already done, False otherwise.
    """
    if os.path.exists(flag_file_path):
        with open(flag_file_path, 'r') as f:
            status = f.read().strip()
            if status == '1':
                return True
    return False

def set_flag_status(flag_file_path, status='1'):
    """ 
    Set the status flag to indicate that the process is done.
    
    Parameters:
    - flag_file: str, path to the flag file
    - status: str, default '1' meaning the process has completed
    """
    if not os.path.exists(flag_file_path):
        print(f"Flag file {flag_file_path} not found. Creating it...")
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(flag_file_path), exist_ok=True)

    with open(flag_file_path, 'w') as f:
        f.write(status)

def reset_flag_status(flag_file_path):
    """ 
    Reset the flag file to indicate that the process has not run.
    
    Parameters:
    - flag_file: str, path to the flag file
    
    The file will only contain '0' after this operation.
    """
    if not os.path.exists(flag_file_path):
        print(f"Flag file {flag_file_path} not found. Creating it...")
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(flag_file_path), exist_ok=True)

    # Overwrite the flag file with '0'
    with open(flag_file_path, 'w') as f:
        f.write('0')
