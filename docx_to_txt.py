import os
from docx import Document
import argparse

def convert_docx_to_txt(input_file, output_file):
    """
    Convert a DOCX file to TXT format
    """
    try:
        # Load the document
        doc = Document(input_file)
        
        # Extract text from paragraphs
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        
        # Write to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(full_text))
            
        print(f"Successfully converted {input_file} to {output_file}")
        
    except Exception as e:
        print(f"Error converting {input_file}: {str(e)}")

def process_folder(input_folder, output_folder):
    """
    Process all DOCX files in the input folder and save TXT files to output folder
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Process each DOCX file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.docx'):
            input_path = os.path.join(input_folder, filename)
            output_filename = os.path.splitext(filename)[0] + '.txt'
            output_path = os.path.join(output_folder, output_filename)
            convert_docx_to_txt(input_path, output_path)

def main():
    # Define input and output folders
    input_folder = r"C:\Users\Felipe Nunez\Documents\Machine Learning Work\JER\codificacion_final\documents\sed"
    output_folder = r"C:\Users\Felipe Nunez\Documents\Machine Learning Work\JER\codificacion_final\assets\input\interviews\txt\sed"
    
    # Process the folders
    process_folder(input_folder, output_folder)

if __name__ == "__main__":
    main()
