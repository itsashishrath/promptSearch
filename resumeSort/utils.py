import os
import json
import google.generativeai as genai
import PyPDF2
import logging
from pdf2image import convert_from_bytes
import pytesseract
import io

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Adjust the path as needed


class WarningLogger:
    def __init__(self):
        self.warnings = []

    def write(self, message):
        if message.strip():  # Only store non-empty messages
            self.warnings.append(message)

class ResumeParser:
    def __init__(self):
        genai.configure(api_key=os.environ["GEMINISTUDIOKEY2"])
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={
                "temperature": 1,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
                "response_mime_type": "application/json",
            }
        )
    
    def extract_text_from_pdf(self, pdf_file):
        """
        Extracts text from a PDF file. If the PDF contains images, it attempts OCR.
        :param pdf_url: URL of the PDF to extract text from
        :return: Extracted text
        """
        try:
            # Fetch PDF content
            # response = requests.get(pdf_url, timeout=10)
            # response.raise_for_status()
            
            # Create warning logger
            warning_logger = WarningLogger()
            logging.getLogger('PyPDF2').handlers = []
            logging.getLogger('PyPDF2').addHandler(logging.StreamHandler(warning_logger))
            
            # Create PDF reader
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Extract text from all pages
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            # Check if any warnings were logged
            if warning_logger.warnings:
                print(f"PDF warnings detected for , using OCR...")
                return self.extract_text_from_image(pdf_file)
            
            # If no text was extracted, try OCR
            if not text.strip():
                print(f"No text found in , using OCR...")
                return self.extract_text_from_image(pdf_file)
            
            return text
            
        except Exception as e:
            print(f"Error extracting text: {e}")
            return ""

    def extract_text_from_image(self, pdf_file):
        """
        Function to extract text from a PDF containing scanned images using OCR.
        :param pdf_file: PDF file object or bytes-like object containing the PDF content
        :return: Extracted text from the PDF
        """
        try:
            # Check if pdf_file is a file-like object or raw bytes
            if isinstance(pdf_file, bytes):
                # If it's raw bytes, process it directly
                pdf_content = pdf_file
            else:
                # If it's a file-like object, read the content as bytes
                pdf_content = pdf_file.read()

            # Convert PDF to images using convert_from_bytes (since the PDF content is in memory)
            images = convert_from_bytes(pdf_content)

            # Apply OCR to each image and extract text
            text = ""
            for img in images:
                text += pytesseract.image_to_string(img) + "\n"

            return text
        except Exception as e:
            print(f"Error during OCR processing: {e}")
            return ""

    def _validate_parsed_data(self, data):
        required_fields = ["personalInfo", "education", "workExperience", "skills"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
        return data

    def parse_resume(self, pdf_file):
        text = self.extract_text_from_pdf(pdf_file)

        structure = """
                    {
                        "personalInfo": {
                            "firstName": "string",
                            "lastName": "string",
                            "email": "string",
                            "mobile_number": "string",
                            "location": {
                            "city": "string",
                            "state": "string",
                            "country": "string"
                            },
                            "linkedIn": "string",
                            "portfolio": "string",
                            "workAuthorization": "string"
                        },
                        "professionalSummary": "string",
                        "education": [
                            {
                            "bachelors_course": "string",
                            "bachelors_specialization": "string",
                            "bachelors_institute": "string",
                            "bachelors_year_of_graduation": "year_only",
                            "bachelors_percentage": "number",

                            "masters_course": "string",
                            "masters_specialization": "string",
                            "masters_institute": "string",
                            "masters_year_of_graduation": "year_only"
                            "masters_percentage" : "number",
                            }
                        ],
                        "workExperience": [
                            {
                            "company": "string",
                            "jobTitle": "string",
                            "location": "string",
                            "start": "month_year_only(MMYYYY)",
                            "end": "month_year_only(MMYYYY)",
                            "responsibilities": [
                                "string"
                            ],
                            "technologiesUsed": [
                                "string"
                            ]
                            }
                        ],
                        "skills": [
                            "string"
                        ],
                        "projects": [
                            {
                            "name": "string",
                            "description": "string",
                            "startDate": "date",
                            "endDate": "date",
                            "technologiesUsed": [
                                "string"
                            ],
                            "url": "string",
                            "achievements": [
                                "string"
                            ]
                            }
                        ],
                        "awardsAndAchievements": [
                            {
                            "name": "string",
                            "issuer": "string",
                            "date": "date",
                            "description": "string"
                            }
                        ],
                        "certificates": [
                            {
                            "Title": "string",
                            "Issuer": "string"
                            }
                        ],
                        "otherDetails": [
                            "string"
                        ]
                        }
                    """
        prompt = """
                    Extract information from the following resume text and format it according to the provided JSON structure.
                    Rules:
                    - Fill all fields that can be found in the resume
                    - Use "unavailable" for missing information
                    - Ensure dates are in YYYY-MM-DD format
                    - Numbers should be parsed as actual numbers, not strings
                    - Keep the original case for proper nouns (names, companies, etc.)
                    """
        
        input_prompt = f"{prompt} Resume Text:{text} Output Structure:{structure}"

        try:
            response = self.model.generate_content(
                input_prompt
            )
            parsed_json = json.loads(response.text)
            print(response.text)
            return self._validate_parsed_data(parsed_json)
        except json.JSONDecodeError:
            raise ValueError("Model response was not valid JSON")
        except Exception as e:
            raise Exception(f"Error parsing resume: {str(e)}")


from typing import Dict, List, Any
from datetime import datetime
import re
from django.db.models import QuerySet
from .models import Resume, ResumeDetail, Candidate
import google.generativeai as genai
import os
import json

class ResumeRankingService:
    def __init__(self):
        # Initialize Gemini AI
        genai.configure(api_key=os.environ["GEMINISTUDIOKEY2"])
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={
                "temperature": 1,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
                "response_mime_type": "application/json",
            }
        )
        
        # System prompt for job requirement analysis
        self.system_prompt = """You are a recruitment assistant that helps convert job requirements into structured search criteria. 
        Your task is to analyze the job requirement text and extract key requirements into a structured JSON format.

        Rules:
        1. Convert all requirements into specific, searchable criteria
        2. Expand abbreviated terms (e.g., 'ML' to 'Machine Learning')
        3. Include similar/alternative terms for technologies
        4. Suggest additional relevant skills/requirements if they're commonly associated
        5. For any ambiguous requirements, suggest clarifications

        Output should be in the following JSON structure:
        {
            "requirements": {
                "skills": {
                    "must_have": ["list of required skills"],
                    "nice_to_have": ["list of preferred skills"],
                    "suggested": ["list of relevant skills not mentioned but recommended"]
                },
                "education": {
                    "degrees": ["acceptable degrees"],
                    "minimum_cgpa": number,
                    "preferred_universities_tier": ["tier1", "tier2", etc]
                },
                "experience": {
                    "minimum_years": number,
                    "preferred_years": number,
                    "relevant_roles": ["list of relevant job titles"],
                    "domains": ["list of relevant industry domains"]
                },
                "location": {
                    "cities": ["list of cities"],
                    "remote_ok": boolean
                }
            }
        }"""

    def get_requirements_from_prompt(self, job_prompt: str) -> Dict[str, Any]:
        """Generate structured requirements from job description prompt"""
        response = self.model.generate_content(self.system_prompt + " " + job_prompt)
        try:
            return json.loads(response.text)["requirements"]
        except json.JSONDecodeError:
            raise ValueError("Invalid response from language model")

    def rank_resumes(self, job_prompt: str) -> List[Dict[str, Any]]:
        """Rank all resumes in the database based on job requirements"""
        
        # Debug: print the job prompt to ensure it is passed correctly
        print(f"Job Prompt: {job_prompt}")

        # Get requirements from prompt
        requirements = self.get_requirements_from_prompt(job_prompt)
        
        # Debug: print the parsed requirements to verify the job requirements extraction
        print(f"Parsed Requirements: {requirements}")
        
        # Create scorer instance
        scorer = ResumeScorer(requirements)

        # Get all resume details from the database with related data
        resume_details = ResumeDetail.objects.select_related(
            'resume', 
            'resume__candidate'
        ).all()
        
        # Debug: print the resume details fetched from the database
        print(f"Fetched Resume Details: {len(resume_details)} resumes found.")
        for detail in resume_details:
            print(f"Resume for {detail.resume.candidate.username}, File Name: {detail.resume.file_name}")
        
        # Score and rank resumes
        scored_resumes = []
        for detail in resume_details:
            # Debug: print current resume being processed
            print(f"Processing resume for {detail.resume.candidate.username}, File: {detail.resume.file_name}")

            # Calculate score
            score = scorer.calculate_total_score(detail)
            
            # Debug: print the calculated score for this resume
            print(f"Calculated Score: {score} for resume {detail.resume.file_name}")
            
            # Get individual match details for debugging
            skills_score = scorer.score_skills(detail.skills)
            education_score = scorer.score_education(detail.education)
            experience_score = scorer.score_experience(detail.work_experience)
            location_score = scorer.score_location(detail.personal_info.get('location', {}))

            # Debug: print each score breakdown
            print(f"  Skills Score: {skills_score}")
            print(f"  Education Score: {education_score}")
            print(f"  Experience Score: {experience_score}")
            print(f"  Location Score: {location_score}")
            
            # Append result
            scored_resumes.append({
                'id': str(detail.resume.id),
                'candidate_username': detail.resume.candidate.username,
                'score': score,
                'file_name': detail.resume.file_name,
                'match_details': {
                    'skills_score': skills_score,
                    'education_score': education_score,
                    'experience_score': experience_score,
                    'location_score': location_score
                }
            })
        
        # Debug: print the list of scored resumes before sorting
        print(f"Scored Resumes before sorting: {scored_resumes}")
        
        # Sort by score in descending order
        sorted_resumes = sorted(scored_resumes, key=lambda x: x['score'], reverse=True)

        # Debug: print the sorted resumes
        print(f"Sorted Resumes: {sorted_resumes}")
        
        return sorted_resumes



class ResumeScorer:
    """Class to score resumes based on job requirements"""
    
    def __init__(self, requirements: Dict[str, Any]):
        self.requirements = requirements
        self.weights = {
            'skills': 0.4,
            'education': 0.2,
            'experience': 0.3,
            'location': 0.1
        }

    def score_skills(self, resume_skills: Dict[str, Any]) -> float:
        """
        Score candidate's skills against requirements
        
        Args:
            resume_skills: Dictionary containing candidate's skills
            
        Returns:
            float: Score between 0 and 1
        """
        if not resume_skills or not isinstance(resume_skills, (dict, list)):
            return 0

        # Extract all skills from resume
        all_resume_skills = []
        if isinstance(resume_skills, dict):
            for key, skills in resume_skills.items():
                if isinstance(skills, list):
                    all_resume_skills.extend(skills)
        elif isinstance(resume_skills, list):
            all_resume_skills = resume_skills
            
        if not all_resume_skills:
            return 0

        # Get required skills from requirements
        must_have_skills = set(self.requirements['skills']['must_have'])
        nice_to_have_skills = set(self.requirements['skills']['nice_to_have'])

        # Convert all skills to lowercase for better matching
        resume_skills_set = set(skill.lower() for skill in all_resume_skills)
        
        # Score must-have skills (higher weight)
        must_have_matches = sum(1 for skill in must_have_skills 
                              if any(skill.lower() in resume_skill 
                                    for resume_skill in resume_skills_set))
        must_have_score = must_have_matches / len(must_have_skills) if must_have_skills else 1

        # Score nice-to-have skills (lower weight)
        nice_matches = sum(1 for skill in nice_to_have_skills 
                         if any(skill.lower() in resume_skill 
                               for resume_skill in resume_skills_set))
        nice_to_have_score = nice_matches / len(nice_to_have_skills) if nice_to_have_skills else 1

        # Calculate weighted score
        print('scoring skills')
        return (must_have_score * 0.7 + nice_to_have_score * 0.3)

    def score_education(self, education: List[Dict[str, Any]]) -> float:
        """
        Score candidate's education against requirements

        Args:
            education: List containing a single dictionary with education details.
                Each item in the list should be a dictionary with fields like 'bachelors_course',
                'bachelors_institute', etc.
            
        Returns:
            float: Score between 0 and 1
        """
        # Check if education is a list with at least one dictionary
        if isinstance(education, list) and len(education) > 0:
            # Extract the first dictionary from the list
            education = education[0]
        else:
            return 0  # Return 0 if the list is empty or not in the expected format

        print(f"Education Details: {education}")

        # Define degree keywords
        bachelor_keywords = ['bachelor', 'btech', 'bsc', 'bachelor of technology', 'bachelor of science']
        master_keywords = ['master', 'msc', 'mtech', 'mba', 'master of science', 'master of technology']
        
        # Extract education details
        bachelors_details = {
            'degree': education.get('bachelors_course', ''),
            'institute': education.get('bachelors_institute', ''),
            'percentage': education.get('bachelors_percentage', 0)
        }
        print(f'bachelors_details: {bachelors_details}')

        masters_details = {
            'degree': education.get('masters_course', ''),
            'institute': education.get('masters_institute', ''),
            'percentage': education.get('masters_percentage', 0)
        }
        print(f'masters_details: {masters_details}')

        # Determine highest relevant degree
        relevant_education = None
        if masters_details['degree'] and any(keyword in masters_details['degree'].lower() 
                                            for keyword in master_keywords):
            relevant_education = masters_details
        elif bachelors_details['degree'] and any(keyword in bachelors_details['degree'].lower() 
                                                for keyword in bachelor_keywords):
            relevant_education = bachelors_details

        if not relevant_education:
            return 0

        # Score percentage/CGPA
        min_percentage = self.requirements['education'].get('minimum_cgpa', 0) * 10  # Convert CGPA to percentage
        percentage_score = min(1.0, relevant_education['percentage'] / min_percentage) if min_percentage > 0 else 1.0

        # Score university tier
        uni_score = 0
        if relevant_education['institute']:
            uni_name = relevant_education['institute'].lower()
            tier1_keywords = ['iit', 'nit', 'bits', 'iiit']  # Add more as needed
            uni_score = 1.0 if any(keyword in uni_name for keyword in tier1_keywords) else 0.5
        
        print('scoring education')
        return (percentage_score * 0.6 + uni_score * 0.4)


    def score_experience(self, experience: List[Dict[str, Any]]) -> float:
        """
        Score candidate's experience against requirements
        
        Args:
            experience: List of work experience details
            
        Returns:
            float: Score between 0 and 1
        """
        if not experience or not isinstance(experience, list):
            return 0
        
        total_years = 0
        relevant_years = 0
        
        for exp in experience:
            if not isinstance(exp, dict):
                continue
            
            # Calculate duration
            start_date = exp.get('startDate', '')
            end_date = exp.get('endDate', '')
            
            try:
                start = datetime.strptime(start_date, '%Y-%m-%d')
                end = datetime.strptime(end_date, '%Y-%m-%d') if end_date else datetime.now()
                duration = (end - start).days / 365
                total_years += duration

                # Check role relevancy
                role = exp.get('jobTitle', '').lower()
                required_roles = [r.lower() for r in self.requirements['experience']['relevant_roles']]
                if any(req_role in role for req_role in required_roles):
                    relevant_years += duration
            except (ValueError, TypeError):
                continue
        
        # Calculate scores
        min_years = self.requirements['experience'].get('minimum_years', 0)
        pref_years = self.requirements['experience'].get('preferred_years', min_years)
        
        years_score = min(1.0, total_years / min_years) if min_years > 0 else 1.0
        relevancy_score = relevant_years / total_years if total_years > 0 else 0

        print('scoring experience')
        return (years_score * 0.7 + relevancy_score * 0.3)

    def score_location(self, location: Dict[str, Any]) -> float:
        """
        Score candidate's location against requirements
        
        Args:
            location: Dictionary containing location details
            
        Returns:
            float: Score between 0 and 1
        """
        if not location:
            return 0
        
        required_cities = [city.lower() for city in self.requirements['location']['cities']]
        candidate_city = location.get('city', '').lower()
        
        # Direct city match
        if candidate_city in required_cities:
            return 1.0
        
        # Remote consideration
        if self.requirements['location'].get('remote_ok', False):
            return 0.8
        
        # Willing to relocate (assuming this field exists in your data)
        if location.get('isWillingToRelocate', False):
            return 0.7
        
        return 0.0

    def calculate_total_score(self, resume_detail: ResumeDetail) -> float:
        """
        Calculate total score for a resume
        
        Args:
            resume_detail: ResumeDetail instance from database
            
        Returns:
            float: Total weighted score between 0 and 1
        """
        scores = {
            'skills': self.score_skills(resume_detail.skills),
            'education': self.score_education(resume_detail.education),
            'experience': self.score_experience(resume_detail.work_experience),
        }
        
        print('total scoring')
        return sum(scores[key] * self.weights[key] for key in scores)
