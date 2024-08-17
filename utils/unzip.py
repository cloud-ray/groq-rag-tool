import zipfile

def extract_zip_file(zip_file_path, extract_dir):
    """
    Extracts the contents of a ZIP file to a specified directory.

    Args:
        zip_file_path (str): The path to the ZIP file.
        extract_dir (str): The directory where the contents will be extracted.
    """
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

# Example usage
zip_file_path = 'assets/TO_USE-20240816T144947Z-001.zip'
extract_dir = 'extracted_contents'

extract_zip_file(zip_file_path, extract_dir)