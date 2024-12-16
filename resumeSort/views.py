from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from .models import *
from .serializers import *
from .utils import *
from rest_framework.views import APIView
from django.core.exceptions import ValidationError
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.shortcuts import get_object_or_404

from rest_framework import status
from rest_framework.response import Response
from .models import Resume
from django.shortcuts import get_object_or_404

class CandidateViewSet(viewsets.ModelViewSet):
    queryset = Candidate.objects.all()
    serializer_class = CandidateSerializer
    lookup_field = 'username'

    @action(detail=True, methods=['POST'])
    def upload_resume(self, request, username=None):
        candidate = self.get_object()

        if 'file' not in request.FILES:
            return Response(
                {'error': 'No file provided'},
                status=status.HTTP_400_BAD_REQUEST
            )

        resume_file = request.FILES['file']
        if not resume_file.name.endswith('.pdf'):
            return Response(
                {'error': 'Only PDF files are supported'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Check if the candidate already has a resume
        existing_resumes = Resume.objects.filter(candidate=candidate)
        if existing_resumes.exists():
            # Delete the old resumes (keeping only the latest one)
            existing_resumes.delete()

        # Create a new resume instance for the candidate
        resume = Resume.objects.create(
            candidate=candidate,
            file_name=resume_file.name,
            original_file=resume_file
        )

        # Parse the new resume
        parser = ResumeParser()
        parsed_data = parser.parse_resume(resume_file)

        if parsed_data:
            ResumeDetail.objects.create(
                resume=resume,
                personal_info=parsed_data.get('personalInfo'),
                professional_summary=parsed_data.get('professionalSummary'),
                education=parsed_data.get('education'),
                work_experience=parsed_data.get('workExperience'),
                skills=parsed_data.get('skills'),
                projects=parsed_data.get('projects'),
                awards_achievements=parsed_data.get('awards and Achievements'),
                certificates=parsed_data.get('certificates'),
                other_details=parsed_data.get('Other Details mentioned')
            )

        # Serialize and return the new resume
        serializer = ResumeSerializer(resume)
        return Response(serializer.data, status=status.HTTP_201_CREATED)

import logging

class RankResumesView(APIView):
    def post(self, request):
        """
        Rank resumes based on job requirements prompt
        
        Request body:
        {
            "job_prompt": "string"  // Job requirements description
        }
        """
        try:
            job_prompt = request.data.get('job_prompt')
            if not job_prompt:
                return Response(
                    {"error": "job_prompt is required"}, 
                    status=status.HTTP_400_BAD_REQUEST
                )

            ranking_service = ResumeRankingService()
            ranked_resumes = ranking_service.rank_resumes(job_prompt)
            
            serializer = RankedResumeSerializer(ranked_resumes, many=True)
            return Response(serializer.data)

        except ValidationError as e:
            logging.error(f"Validation error: {str(e)}")
            return Response(
                {"error": str(e)}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
            return Response(
                {"error": "Internal server error"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
