import json
import os
from pathlib import Path
import logging
import PyPDF2
import re
from tqdm import tqdm
import uuid

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ResumeTextExtractor:
    def __init__(self, input_dir="downloaded_resumes", output_file="resume_dataset.json"):
        """
        Initialize the text extractor.
        
        Args:
            input_dir (str): Directory containing downloaded resumes
            output_file (str): Output JSON file path
        """
        self.input_dir = Path(input_dir)
        self.output_file = Path(output_file)
        
    def clean_text(self, text):
        """
        Clean and normalize extracted text.
        
        Args:
            text (str): Raw text from PDF
            
        Returns:
            str: Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove numbers (optional, comment out if you want to keep numbers)
        # text = re.sub(r'\d+', '', text)
        
        return text.strip()
    
    def extract_text_from_pdf(self, pdf_path):
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path (Path): Path to PDF file
            
        Returns:
            tuple: (success (bool), text (str), error_message (str))
        """
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract text from all pages
                for page in pdf_reader.pages:
                    text += page.extract_text() + " "
            
            # Clean the extracted text
            cleaned_text = self.clean_text(text)
            
            return True, cleaned_text, None
            
        except Exception as e:
            error_msg = f"Error extracting text from {pdf_path}: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg
    
    def create_dataset(self):
        """
        Process all PDFs and create a dataset.
        
        Returns:
            dict: Results containing successful and failed extractions
        """
        results = {
            'documents': [],
            'failed': []
        }
        
        # Get all PDF files
        pdf_files = list(self.input_dir.glob('*.pdf'))
        
        # Process each PDF file
        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            # Generate a unique ID for the document
            doc_id = str(uuid.uuid4())
            
            success, text, error_msg = self.extract_text_from_pdf(pdf_path)
            
            if success:
                document = {
                    'id': doc_id,
                    'filename': pdf_path.name,
                    'text': text
                }
                results['documents'].append(document)
            else:
                results['failed'].append({
                    'filename': pdf_path.name,
                    'error': error_msg
                })
        
        return results

def main():
    # Initialize extractor
    extractor = ResumeTextExtractor()
    
    # Process PDFs and create dataset
    logger.info("Starting text extraction from PDFs...")
    results = extractor.create_dataset()
    
    # Print summary
    logger.info("\nExtraction Summary:")
    logger.info(f"Successfully processed: {len(results['documents'])} documents")
    logger.info(f"Failed extractions: {len(results['failed'])}")
    
    # Save dataset to JSON file
    with open(extractor.output_file, 'w', encoding='utf-8') as f:
        json.dump(results['documents'], f, indent=4, ensure_ascii=False)
    
    logger.info(f"\nDataset saved to {extractor.output_file}")
    
    # Save failed extractions to a separate file if any
    if results['failed']:
        failed_file = 'failed_extractions.json'
        with open(failed_file, 'w') as f:
            json.dump(results['failed'], f, indent=4)
        logger.info(f"Failed extractions saved to {failed_file}")
    
if __name__ == "__main__":
    main()