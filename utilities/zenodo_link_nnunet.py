import os

def create_user_file():
    # Ask the user for username and password
    filename = input("Enter UNETR filename: ")
    zenodo_link = input("Enter UNETR doi: ")

    # Create a text file and write the username and password to it
    with open("zenodo_nnunet.txt", "w") as file:
        file.write(f"Filename: {filename}\n")
        file.write(f"doi: {zenodo_link}\n")

def read_user_file():
    # Read the username and password from the text file
    with open("zenodo_nnunet.txt", "r") as file:
        lines = file.readlines()
        filename = lines[0].split(":")[1].strip()
        doi = lines[1].split(":")[1].strip()

    return filename, doi

def main():
    # Check if the file exists
    if os.path.exists("zenodo_nnunet.txt"):
        # If the file exists, read username and password
        existing_filename, existing_zenodo_link = read_user_file()
    else:
        # If the file does not exist, create a new file and ask for username and password
        create_user_file()
        print("Zenodo file created successfully.")
        existing_filename, existing_zenodo_link = read_user_file()
    return existing_filename, existing_zenodo_link