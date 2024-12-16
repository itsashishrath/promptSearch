# models.py
from django.db import models
import uuid

class Candidate(models.Model):
    username = models.CharField(max_length=255, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.username

class Resume(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    candidate = models.ForeignKey(Candidate, on_delete=models.CASCADE, related_name='resumes')
    file_name = models.CharField(max_length=255)
    original_file = models.FileField(upload_to='resumes/')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.candidate.username}'s resume - {self.file_name}"

class ResumeDetail(models.Model):
    resume = models.OneToOneField(Resume, on_delete=models.CASCADE, related_name='details')
    personal_info = models.JSONField(null=True, blank=True)
    professional_summary = models.TextField(null=True, blank=True)
    education = models.JSONField(null=True, blank=True)
    work_experience = models.JSONField(null=True, blank=True)
    skills = models.JSONField(null=True, blank=True)
    projects = models.JSONField(null=True, blank=True)
    awards_achievements = models.JSONField(null=True, blank=True)
    certificates = models.JSONField(null=True, blank=True)
    other_details = models.JSONField(null=True, blank=True)

    def __str__(self):
        return f"Details for {self.resume.candidate.username}'s resume"
