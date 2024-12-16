# serializers.py
from rest_framework import serializers
from .models import *

from rest_framework import serializers

class ResumeDetailSerializer(serializers.ModelSerializer):
    class Meta:
        model = ResumeDetail
        exclude = ('resume',)

class ResumeSerializer(serializers.ModelSerializer):
    details = ResumeDetailSerializer(read_only=True)
    
    class Meta:
        model = Resume
        fields = ['id', 'file_name', 'created_at', 'details']

class CandidateSerializer(serializers.ModelSerializer):
    resumes = ResumeSerializer(many=True, read_only=True)
    
    class Meta:
        model = Candidate
        fields = ['username', 'created_at', 'updated_at', 'resumes']

class RankedResumeSerializer(serializers.Serializer):
    id = serializers.UUIDField()
    candidate_username = serializers.CharField()
    score = serializers.FloatField()
    file_name = serializers.CharField()

    class Meta:
        fields = ['id', 'candidate_username', 'score', 'file_name']

