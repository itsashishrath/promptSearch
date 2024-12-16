from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import *

router = DefaultRouter()
router.register(r'candidates', CandidateViewSet)

urlpatterns = [
    path('api/', include(router.urls)),
    path('api/rank-resumes/', RankResumesView.as_view(), name='rank-resumes'),

]